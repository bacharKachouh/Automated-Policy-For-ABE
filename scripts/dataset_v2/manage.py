#!/usr/bin/env python3
"""Dataset v2 labelling pipeline.

Subcommands:
  emit   - sample N unlabeled rows (with text-hash dedup) and write JSON for labelling
  ingest - read labelled JSON back, append to medical_data_v2.csv
  stats  - show progress vs the per-tier targets

CSV columns: source, source_idx, text, label, trigger, confidence, text_hash

text_hash dedups across sources: once a piece of text is labelled, we never
emit a near-duplicate from any source.
"""
import argparse, csv, hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "training" / "data" / "medical_data_v2.csv"
RAW_DIR = ROOT / "training" / "data" / "raw"
COLS = ["source", "source_idx", "text", "label", "trigger", "confidence", "text_hash"]
VALID_LABELS = {"HC", "C", "R", "P"}
TARGETS = {"HC": 25000, "C": 25000, "R": 25000, "P": 25000}

_WS = re.compile(r"\s+")


def _normalise(text):
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    t = _WS.sub(" ", t)
    return t


def _text_hash(text):
    return hashlib.sha1(_normalise(text).encode()).hexdigest()


# ---------- source loaders: each yields (source_idx, text) ----------

def _load_agbonnet():
    path = RAW_DIR / "augmented_notes_30K.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("note") or row.get("full_note") or ""
            yield str(row.get("idx", "")), text


def _load_asclepius():
    path = RAW_DIR / "asclepius_clinical_notes.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    import pandas as pd
    df = pd.read_parquet(path)
    text_col = "patient" if "patient" in df.columns else "note"
    for i, row in df.iterrows():
        yield f"asc_{i}", row[text_col]


def _load_timschopf():
    p1 = RAW_DIR / "data" / "train-00000-of-00001.parquet"
    p2 = RAW_DIR / "medical_abstracts_train.parquet"
    path = p1 if p1.exists() else p2
    if not path.exists():
        raise FileNotFoundError(path)
    import pandas as pd
    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        yield f"ts_{i}", row["medical_abstract"]


def _load_publichealth():
    path = RAW_DIR / "publichealth_bulletins.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        for i, line in enumerate(f):
            yield f"ph_{i}", json.loads(line)["text"]


SOURCES = {
    "agbonnet": _load_agbonnet,
    "asclepius": _load_asclepius,
    "timschopf": _load_timschopf,
    "publichealth": _load_publichealth,
}


# ---------- CSV state ----------

def _ensure_csv():
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(COLS)
        return
    with open(CSV_PATH, newline="") as f:
        first = next(csv.reader(f), None)
    if first and "text_hash" not in first:
        import shutil
        shutil.copy(CSV_PATH, str(CSV_PATH) + ".bak")
        with open(str(CSV_PATH) + ".bak", newline="") as fin, open(CSV_PATH, "w", newline="") as fout:
            r = csv.DictReader(fin)
            w = csv.DictWriter(fout, fieldnames=COLS)
            w.writeheader()
            for row in r:
                row.setdefault("text_hash", _text_hash(row.get("text", "")))
                w.writerow({c: row.get(c, "") for c in COLS})


def _load_state():
    _ensure_csv()
    labeled_keys, seen_hashes = set(), set()
    label_counts = {l: 0 for l in VALID_LABELS}
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            labeled_keys.add((r["source"], r["source_idx"]))
            if r.get("text_hash"):
                seen_hashes.add(r["text_hash"])
            if r["label"] in label_counts:
                label_counts[r["label"]] += 1
    return labeled_keys, seen_hashes, label_counts


def _append(rows):
    _ensure_csv()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        for r in rows:
            w.writerow(r)


# ---------- commands ----------

def cmd_emit(args):
    if args.source not in SOURCES:
        sys.exit(f"Unknown source {args.source!r}. Choices: {list(SOURCES)}")
    labeled_keys, seen_hashes, _ = _load_state()
    src_labeled = {idx for src, idx in labeled_keys if src == args.source}

    candidates, skipped_dup = [], 0
    for idx, text in SOURCES[args.source]():
        if idx in src_labeled:
            continue
        wc = len(text.split())
        if not (args.min_words <= wc <= args.max_words):
            continue
        h = _text_hash(text)
        if h in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(h)
        candidates.append({"idx": idx, "text": text, "text_hash": h})
        if len(candidates) >= args.batch * 3:
            break

    random.seed(args.seed)
    random.shuffle(candidates)
    batch = candidates[:args.batch]
    if not batch:
        sys.exit(f"No unlabeled rows match the filter for source {args.source}")

    Path(args.out).write_text(json.dumps({
        "source": args.source,
        "batch_size": len(batch),
        "rows": batch,
        "skipped_dup": skipped_dup,
    }, indent=1))
    print(f"Wrote {len(batch)} unlabeled rows from {args.source} -> {args.out}")
    if skipped_dup:
        print(f"  (skipped {skipped_dup} duplicates by text-hash)")


def cmd_ingest(args):
    data = json.loads(Path(args.path).read_text())
    rows = data.get("rows") or data
    if not isinstance(rows, list):
        sys.exit("Input must be a list or {rows:[...]} of label objects.")
    source = data.get("source") if isinstance(data, dict) else None

    labeled_keys, seen_hashes, _ = _load_state()
    out_rows, skipped = [], 0
    for r in rows:
        idx = str(r.get("idx", ""))
        src = r.get("source", source)
        if not src or not idx:
            skipped += 1; continue
        if (src, idx) in labeled_keys:
            skipped += 1; continue
        label = (r.get("label") or "").strip().upper()
        if label not in VALID_LABELS:
            skipped += 1; continue
        text = r.get("text", "")
        h = r.get("text_hash") or _text_hash(text)
        if h in seen_hashes:
            skipped += 1; continue
        seen_hashes.add(h)
        out_rows.append({
            "source": src, "source_idx": idx, "text": text,
            "label": label,
            "trigger": (r.get("trigger") or "").strip(),
            "confidence": (r.get("confidence") or "high").strip().lower(),
            "text_hash": h,
        })
    _append(out_rows)
    print(f"Appended {len(out_rows)} rows; skipped {skipped}")


def cmd_stats(args):
    _, _, label_counts = _load_state()
    from collections import Counter
    src, both, n = Counter(), Counter(), 0
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            src[r["source"]] += 1
            both[(r["source"], r["label"])] += 1
    print(f"Total labelled rows: {n}")
    print(f"By source: {dict(src)}")
    print()
    print("Per-tier progress:")
    for lab in ["HC", "C", "R", "P"]:
        c = label_counts.get(lab, 0)
        t = TARGETS[lab]
        pct = 100.0 * c / t if t else 0
        bar = "#" * int(pct / 5)
        print(f"  {lab:3s}  {c:6d} / {t:6d}  [{bar:<20s}]  {pct:5.1f}%")
    print()
    print("By (source, label):")
    for (s, l), c in sorted(both.items()):
        print(f"  {s:14s} {l:3s}  {c}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("--source", required=True)
    pe.add_argument("--batch", type=int, default=200)
    pe.add_argument("--out", default="/tmp/batch.json")
    pe.add_argument("--min-words", type=int, default=80)
    pe.add_argument("--max-words", type=int, default=500)
    pe.add_argument("--seed", type=int, default=42)
    pe.set_defaults(func=cmd_emit)
    pi = sub.add_parser("ingest")
    pi.add_argument("path")
    pi.set_defaults(func=cmd_ingest)
    ps = sub.add_parser("stats")
    ps.set_defaults(func=cmd_stats)
    args = p.parse_args()
    args.func(args)
