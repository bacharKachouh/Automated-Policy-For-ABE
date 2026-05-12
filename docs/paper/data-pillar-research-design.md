# Data Pillar — Research Design & Paper Defense

**Status:** Draft for review
**Owner:** the user (paper author)
**Last updated:** 2026-05-12

The purpose of this document is to enumerate everything the paper needs to defend the **Data Pillar** of the three-pillar architecture: automated security classification of medical records as the front end of the access-control pipeline. We are not implementing or training anything in this document — it is the checklist that the implementation work in `docs/superpowers/plans/` will be measured against.

---

## 1. What we are claiming

The paper's data pillar story is, roughly:

> *Given a medical record, we can automatically assign a meaningful security classification, and that classification is reliable enough to drive downstream ABE policy generation.*

We have to split that into separate falsifiable claims, because reviewers attack at the claim level.

| # | Claim | What evidence proves it |
|---|---|---|
| **C1** | A fine-tuned domain language model classifies medical text into our 4-tier security scheme with non-trivial accuracy. | Test-set metrics on held-out data: accuracy, macro-F1, per-class F1, confusion matrix. |
| **C2** | The 4-tier scheme is principled, not ad-hoc. | Mapping to existing standards (HIPAA categories, GDPR Art. 9 special-category data, NIST 800-171 CUI tiers, ISO 27001 information classification). Cite. |
| **C3** | Section-level classification + threshold aggregation is a sound document-level decision rule. | Sensitivity analysis over thresholds, comparison against alternative aggregators (majority vote, max-severity, learned aggregator). |
| **C4** | The model output is a usable input to the ABE policy generator. | End-to-end pipeline numbers: % of documents that produce a satisfiable policy, % where the policy correctly excludes / includes test users. |
| **C5** | Our approach beats reasonable baselines. | Reported numbers for at least: TF-IDF + LogReg, vanilla BERT, DistilBERT, BioClinicalBERT (ours), and a zero/few-shot LLM baseline. |
| **C6** | The approach generalises beyond our training data. | Out-of-distribution evaluation on at least one different distribution (e.g. a real public clinical-text corpus, even unlabeled — measured via downstream policy correctness or label-consistency proxy). |

If a reviewer can't point to a paragraph that defends each claim with numbers or argument, the paper isn't ready.

---

## 2. The 4-tier scheme — why these labels

This is one of the easiest things for a reviewer to attack: "Why four levels? Why these names? Where do they map to in the real world?"

**Current labels:** Highly Confidential, Confidential, Restricted, Public.

**Defense to write:**
- Map each tier to one or more recognised standards. Suggested mapping (to be verified against the actual standards):
  - **Highly Confidential** ↔ HIPAA PHI (18 identifiers + diagnosis details), GDPR Art. 9 special-category health data
  - **Confidential** ↔ patient-identifiable but non-special-category data (administrative records with names, billing, prescriptions linked to a patient)
  - **Restricted** ↔ research-context data: aggregated or de-identified but redistribution-controlled
  - **Public** ↔ routine, de-identified, non-sensitive (general wellness summaries, public health bulletins)
- Cite ISO 27001 information classification or organisation-specific frameworks if available.
- **Open question:** is this mapping defensible, or do we need an extra tier or to rename? Decide before training the real model.

**Threat to validity:** the boundaries between tiers are fuzzy (e.g. is a prescription Confidential or Highly Confidential?). The current `medical_data.csv` contains examples that contradict each other (prescriptions appear as both Confidential and Highly Confidential). This will hurt model performance and credibility. The cleanup proposed in §6 must enforce label consistency via written labelling guidelines.

---

## 3. Dataset — the biggest threat to the paper

**Current state of `training/data/medical_data.csv`:**

| Property | Value | Concern |
|---|---|---|
| Rows | 10,475 | Looks fine on the surface |
| Unique texts | 9,091 | **1,384 duplicates (13%)** — train/test leakage almost certain |
| Length | 1–19 words, avg 10.4 | Far shorter than real clinical notes (100s of words) |
| Source | `training/generate_data.py` has 341 hand-coded rows | **10K row CSV is non-reproducible** — we cannot regenerate our own training set |
| Templates | Many near-duplicate "Lab Test: X, Result: Y" lines, all `Confidential` | Model will memorise template; doesn't learn semantics |
| Class balance | C 45 / HC 34 / R 11 / P 10 | Imbalanced, no per-class weighting |
| Labels | Single-author synthetic | No inter-annotator agreement |

**Reviewer attacks we will get:**
1. *"How do we know your model isn't just memorising templates?"* — show de-duplicated split + per-template performance.
2. *"Does it generalise to real medical text?"* — OOD evaluation required.
3. *"Synthetic data is suspicious for healthcare."* — cite synthetic-data papers and run a real-data sanity check, however small.
4. *"Where are your labelling guidelines?"* — write them. Without guidelines you can't even argue your labels are consistent.

**Dataset strategy decisions to make (open):**

- **(a) Defensible synthetic generator.** Replace the static `generate_data.py` with a documented, reproducible LLM-driven generator (we have GPU; or use HF Inference) that produces longer, more varied medical text. Document the prompts, seed, model used, post-filters (dedup, length, label consistency). Cite synthetic data lineage (e.g. *Synthetic Data for Clinical NLP*, *MedTextNN*, *DPM-SOLVER-style augmentation*).
- **(b) Pull a real corpus.** Realistic candidates:
  - **MIMIC-III/IV discharge summaries** — PhysioNet credentialing required (5-10 days); free.
  - **n2c2 / i2b2 challenges** — clinical NLP shared tasks; some have sensitivity-adjacent labels.
  - **MTSamples** — public transcribed medical reports, no labels for our task but useful for OOD.
  - **PubMed abstracts** — published research text, defensible Restricted/Public label proxy.
- **(c) Hybrid.** Train on rich synthetic; OOD-evaluate on real corpus.

**Recommendation to bake into the paper:** (a) + (c). Build a defensible synthetic dataset *and* show that a model trained on it transfers reasonably to real text. This is the path that minimises legal/IRB friction while still standing up to reviewers.

**Concrete deliverables for the dataset rebuild:**
- `training/generate_dataset.py` — fully reproducible from a seed, with explicit prompts, model name, post-filters.
- `training/labelling_guidelines.md` — written rules for assigning each tier, with positive and negative examples.
- `training/data/medical_data_v2.csv` — clean, de-duplicated, balanced where reasonable, length-filtered (≥ 30 words).
- `training/data/medical_data_v2_ood.csv` — small OOD slice from a real corpus for transfer evaluation.
- A short `training/data/README.md` describing how the dataset was generated, version history, and exactly which scripts produce what.

---

## 4. Model — what we train and what we compare against

The paper needs at least 4 numbers in a results table, ideally 5+.

| Model | Why include it |
|---|---|
| **TF-IDF + LogisticRegression** (sklearn) | Non-neural baseline. Strong for short text. Establishes "is the problem even hard?". |
| **DistilBERT (base)** | Small-transformer baseline. Shows whether domain pre-training matters. |
| **Vanilla BERT (base)** | Same architecture as ours but general-domain. Direct ablation of *Bio* contribution. |
| **BioClinicalBERT** (ours) | The proposed model. Headline number. |
| **BioBERT** (optional) | Alternative domain pre-training — strengthens "is BioClinical specifically the right choice?" |
| **Zero/few-shot LLM** (optional but increasingly expected) | Modern baseline. Use a small instruction-tuned LLM (Llama-3.1-8B-Instruct, Mistral-7B-Instruct, Qwen-2.5-7B) zero-shot or with a few labelled examples. Shows we beat a general LLM doing the same task. |

**Hyperparameter ablations to consider (one knob at a time):**
- Max sequence length (128 → 256 → 512). 128 truncates real notes.
- Training data size (25%, 50%, 100%) to demonstrate data-scaling behaviour.
- Class-balanced loss vs vanilla cross-entropy.

**Things the paper should not waste pages on:** learning-rate sweeps, optimizer choices. Cite reasonable defaults from the BioClinicalBERT paper.

---

## 5. Evaluation — what we measure and report

**Primary metrics on the section-level test set:**
- Accuracy
- Macro-F1
- Per-class precision / recall / F1
- Confusion matrix (figure)

**Document-level metrics** (because the pipeline operates on documents, not sections):
- Document accuracy under each candidate aggregator (threshold-based current, majority vote, max-severity, learned)
- Document-level macro-F1
- Threshold sensitivity: sweep each threshold in `CLASSIFICATION_THRESHOLDS` over a reasonable range and plot doc-level F1 vs threshold. Show the chosen value is in a stable region, not a knife-edge.

**Pipeline-level metric:**
- % of documents for which the generated ABE policy is *correct* — i.e. a user with the intended role can decrypt, a user without it cannot. Uses the smoke-test scaffolding already built.

**Splits:**
- **De-duplicated** train/val/test (drop or stratify by near-duplicate group).
- **80/10/10** split with fixed seed.
- **OOD test set** — separate, never used for training.

**Reporting discipline:**
- Fixed seed, repeat at least 3 times, report mean ± std.
- Save predictions to disk for failure analysis.
- All results in `results/<run-name>/results.json` so a paper-table script can pull them.

---

## 6. Threats to validity — explicit list

These should appear in a "Limitations" or "Threats to validity" section of the paper, addressed not hidden.

| Threat | Severity | Defense or mitigation |
|---|---|---|
| Synthetic training data | **High** | Reproducible generator, written labelling guidelines, OOD evaluation on real corpus. |
| Train/test leakage from duplicates | **High** | De-duplicated split, near-duplicate detection (e.g. MinHash) before splitting. |
| Class imbalance | Medium | Stratified split + report per-class F1 + class-weighted training as ablation. |
| 4-tier scheme is ad-hoc | Medium | Map to recognised standards (HIPAA, GDPR, NIST). Defend in §2. |
| Section-level vs doc-level granularity | Medium | Compare aggregators in §5. Justify chosen one. |
| Threshold values are arbitrary | Medium | Sensitivity analysis, report plateau region. |
| Only English | Low | Acknowledge, scope as future work. |
| Single-annotator labels | Medium | Document the labelling rules; consider a small adjudicated subset for IAA. |
| Domain transferability | High | OOD evaluation. Be honest about gap. |
| Model size / compute cost | Low | Report training/inference cost. BioClinicalBERT is small. |

---

## 7. Related work the paper must cite or position against

Non-exhaustive — these are the obvious neighbours.

- **Medical text classification with transformers:** Alsentzer et al. (BioClinicalBERT), Lee et al. (BioBERT), Yang et al. (Med-PaLM).
- **PHI / de-identification:** Stubbs & Uzuner i2b2 de-identification; n2c2 challenges.
- **Synthetic clinical data:** Wang et al. (MedTextNN), Liu et al. (clinical augmentation surveys), Frei & Kramer (PubMedQA synth).
- **Sensitivity classification in records:** thin literature; this is where the paper has space to contribute.
- **Access control / ABE for healthcare:** Akinyele et al. (early ABE-EMR), Liu et al. (cloud EMR), Lewko-Waters (the scheme we use).
- **Document classification with hierarchy / aggregation:** Yang et al. HANs (hierarchical attention networks) for an alternative aggregation scheme.

**Positioning sentence (draft):** *"Prior ABE-for-healthcare work assumes the security label of a record is supplied by hand or by simple rules. We close this gap by treating the label as the output of a trained classifier, evaluate that classifier, and demonstrate the end-to-end pipeline."*

---

## 8. Implementation work needed (separate from this design doc)

These belong in a future plan in `docs/superpowers/plans/`, not here. Listed so we don't forget:

- [ ] Eval harness: train/val/test split with de-duplication, baselines (TF-IDF, DistilBERT, BERT, BioClinicalBERT), metrics, results JSON.
- [ ] Threshold-sweep script.
- [ ] Pipeline-level evaluation hooks (already partially present in `audit_runs`).
- [ ] LLM-driven dataset generator with seed + prompts + filters.
- [ ] Labelling guidelines document.
- [ ] OOD corpus acquisition (MIMIC PhysioNet credentialing — start early; takes ~1-2 weeks).
- [ ] Failure-analysis script (top-N misclassifications per class).

---

## 9. Open questions for the user

These need an explicit decision before we start implementing.

1. ~~4-tier scheme — keep / merge / rename?~~ **DECIDED (2026-05-12):** keep HC / C / R / P; add the HIPAA/GDPR mapping in §2 to the paper. Dataset cleanup must enforce the mapping.
2. ~~Dataset strategy — synthetic-rebuild only, or also pull real data (MIMIC)?~~ **DECIDED (2026-05-12):** scaled corpus (target 50K-100K rows) sourced from public medical text + labeled by Claude through Claude Code sessions (leveraging Max subscription, no per-token API cost). Source candidates: AGBonnet/augmented-clinical-notes (MIT, 30K PMC notes), TimSchopf/medical_abstracts (CC-BY, 14K), PubMed Central case reports (E-utilities). Labels assigned by Claude Opus 4.7 in this Claude Code workflow using HIPAA/GDPR-anchored prompts from §2 (the prompt is the labelling guidelines from Q5). Pipeline: pull source → chunk into ~1.5-3K row batches → label each batch in a Claude Code session → append to CSV → iterate ~20-30 sessions. Quality controls: exact + near-dup dedup (MinHash), length filter (50-300 words), 5-10% re-label sample for inter-method agreement, hand-spot-check ~100 rows. No external data wait (PhysioNet / MIMIC deferred). HC-tier validation against PHI-detection corpora is optional follow-up.
3. ~~LLM baseline — include or skip?~~ **DECIDED (2026-05-12):** include both zero-shot AND few-shot LLM (Qwen-2.5-7B-Instruct or comparable open-weight 7B) as baselines in the main results table. Evaluated on the same test set as the fine-tuned models.
4. ~~Document granularity~~ **DECIDED:** keep section-level + threshold aggregator. Add a threshold-sensitivity analysis + alternative-aggregator ablation (max-severity, majority vote) in evaluation. Hierarchical (HAN) deferred as future work.
5. ~~Labelling guidelines~~ **DECIDED:** v1 written at `training/labelling_guidelines.md`. Doubles as the labelling prompt for Claude. Tightened reproductive/oncology/pediatric triggers based on user spot-check (2026-05-12).
6. ~~Compute budget~~ **DECIDED:** 2× V100 32GB on the trooper server + Claude Max for Claude-Code-driven labelling. Not a constraint.

**Dataset target locked (2026-05-12):** 100K rows, ~25K per tier (HC/C/R/P), balanced. Sources: AGBonnet (PMC notes, HC + C), TimSchopf/medical_abstracts + PubMed E-utilities (R), public-health bulletins + generated content (P). Labelling done in Claude Code sessions via Max sub (no per-token API cost).

Answer these and the next step is a writing-plans pass to turn §8 into a real implementation plan.
