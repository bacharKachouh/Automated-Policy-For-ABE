# Dataset v2 — Labelling Pipeline

Build a balanced 4-tier security-labelled medical-text corpus (`training/data/medical_data_v2.csv`) for the Data Pillar classifier.

**Target:** 25K rows per tier × 4 tiers = **100K rows total**.

**Tiers:** HC (Highly Confidential) / C (Confidential) / R (Restricted) / P (Public). See `training/labelling_guidelines.md` for the rubric.

---

## Sources

| Source key | Provenance | License | Rows available | Best for |
|---|---|---|---|---|
| `agbonnet` | AGBonnet/augmented-clinical-notes — PMC patient summaries + GPT-4 structured extracts | MIT | ~30K (some internal dupes) | HC + C |
| `asclepius` | starmpcc/Asclepius-Synthetic-Clinical-Notes — GPT-3.5-derived from PMC-Patients | CC-BY-NC-SA | 158K unique | HC + C (backup / scaling) |
| `timschopf` | TimSchopf/medical_abstracts — research abstracts, 5 disease classes | CC-BY-SA-3.0 | 11.5K | R (research abstracts) |
| `publichealth` | Seeded public-health bulletins (CDC/MedlinePlus style); grow over time | CC0 (our writing) | 30 seed (extend) | P |

All sources pulled to `training/data/raw/` on the server.

---

## Workflow per labelling session

### 1. Emit a batch

```bash
cd ~/Automated_ABE
source /opt/miniconda/etc/profile.d/conda.sh && conda activate abe

python scripts/dataset_v2/manage.py emit \
    --source agbonnet \
    --batch 200 \
    --out /tmp/batch.json \
    --min-words 80 --max-words 500
```

Produces `/tmp/batch.json` containing 200 unlabeled rows. Already-labelled `(source, source_idx)` pairs are skipped automatically, and any text whose normalised hash matches an existing row is also skipped (cross-source dedup).

### 2. Label inside a Claude Code session

Open a Claude Code conversation. Read the JSON, apply the rubric in `training/labelling_guidelines.md` to each row, and emit the same shape with `label`, `trigger`, `confidence` added per row:

```json
{
  "source": "agbonnet",
  "rows": [
    {"idx": "...", "text": "...", "text_hash": "...",
     "label": "HC", "trigger": "confirmed oncology", "confidence": "high"},
    ...
  ]
}
```

Save the labelled JSON back to the server (e.g. `/tmp/batch_labeled.json`).

### 3. Ingest

```bash
python scripts/dataset_v2/manage.py ingest /tmp/batch_labeled.json
```

Output reports rows appended and rows skipped (already-labelled, duplicate, or invalid label).

### 4. Check progress

```bash
python scripts/dataset_v2/manage.py stats
```

Shows total, per-source, and per-tier counts with progress bars against the 25K-per-tier target.

---

## Realistic throughput per session

Labelling 200 rows of 100-500-word medical text uses non-trivial context. A single Claude Code session can comfortably handle **150-300 rows**. Hitting 100K rows requires many sessions (~30-60+). Recommended pacing: 1-3 sessions per day, mixing sources to keep distribution balanced.

---

## Source rotation strategy

To hit balanced 25K-per-tier targets, sources rotate based on which tier needs filling:

| If you're short on... | Use source... | Why |
|---|---|---|
| HC | agbonnet, then asclepius | Both rich in detailed clinical cases including cancer, genetic, mental health |
| C | agbonnet, asclepius, even timschopf clinical-case abstracts | Standard clinical content is the largest natural category |
| R | timschopf | Built for research-abstract content; expect mostly R after labelling |
| P | publichealth | Hand-curated P-tier; extend the JSONL with more entries as needed |

You can also **pre-bias the emit** by stripping low-relevance candidates. For example, after agbonnet reaches its useful HC quota, switch to asclepius for further HC sourcing.

---

## Adding more sources

To wire up an additional source, add a loader in `manage.py:SOURCES` that yields `(source_idx, text)` tuples and ensure raw data lives under `training/data/raw/`. Source keys are namespaced so a single piece of text appearing in two sources is dedup'd by `text_hash` regardless of which source emitted it first.

---

## Extending the P-tier seed

`training/data/raw/publichealth_bulletins.jsonl` is a starter set of ~30 generated bulletins. Add more entries (one JSON object per line, with a `text` field) when you have time, drawn from:

- CDC fact sheets, advisories, weekly MMWR-style content
- WHO public health bulletins
- NIH MedlinePlus health-topic summaries
- Original Claude-generated P-tier content using the rubric

Aim for ~25K entries here over time, varying topic, length, and style.

---

## File layout

```
training/
  data/
    medical_data_v2.csv          ← the dataset being built
    medical_data_v2.csv.bak      ← auto-backup when schema migrates
    raw/
      augmented_notes_30K.jsonl  ← AGBonnet source
      asclepius_clinical_notes.parquet
      data/train-00000-of-00001.parquet  ← TimSchopf
      publichealth_bulletins.jsonl
  labelling_guidelines.md        ← the rubric
scripts/dataset_v2/
  manage.py                      ← the pipeline
  README.md                      ← this file
```

---

## Common operations

```bash
# How balanced am I?
python scripts/dataset_v2/manage.py stats

# How many AGBonnet rows are still unlabeled, with 80-500 word filter?
python -c "
from scripts.dataset_v2.manage import SOURCES, _load_state
ks, _, _ = _load_state()
src_labeled = {idx for s, idx in ks if s == 'agbonnet'}
n = sum(1 for idx, t in SOURCES['agbonnet']()
        if idx not in src_labeled and 80 <= len(t.split()) <= 500)
print(f'unlabeled agbonnet rows in length window: {n}')
"

# Export final CSV without text (smaller, for paper)
python -c "
import csv
with open('training/data/medical_data_v2.csv') as f, \
     open('training/data/medical_data_v2_meta.csv', 'w', newline='') as out:
    r = csv.DictReader(f); w = csv.DictWriter(out, ['source','source_idx','label','trigger','confidence','text_hash'])
    w.writeheader()
    for row in r:
        w.writerow({k: row[k] for k in w.fieldnames})
"
```
