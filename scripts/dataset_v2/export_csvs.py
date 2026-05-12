#!/usr/bin/env python3
"""
Export each source as a labelling-ready CSV.

Output: training/data/for_labelling/<source>.csv

Each CSV has columns:
  source, source_idx, text, word_count, label, trigger, confidence, text_hash

`label`, `trigger`, `confidence` are empty — fill them in then `ingest` the
file (the existing pipeline handles CSV → CSV ingestion via a small wrapper
in this module).

Filters applied at export time:
- length window per source (defaults: 80-500 words for clinical, 50-300 for bulletins)
- exact-text dedup by `text_hash` (cross-source: a piece of text labelled
  in any earlier export is dropped from later exports)
- skips rows already present in medical_data_v2.csv

After labelling, run:
    python scripts/dataset_v2/export_csvs.py ingest <path-to-filled.csv>
"""
import argparse, csv, hashlib, json, re, sys
from pathlib import Path

# Reuse the loaders + dedup from manage.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.dataset_v2.manage import (
    SOURCES, _text_hash, _load_state, _append, CSV_PATH, VALID_LABELS,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "training" / "data" / "for_labelling"
OUT_COLS = ["source", "source_idx", "text", "word_count",
            "label", "trigger", "confidence", "text_hash"]

# Per-source length windows + optional row caps
SOURCE_CONFIG = {
    "agbonnet":     {"min_words":  80, "max_words": 500, "cap": None},
    "asclepius":    {"min_words":  80, "max_words": 500, "cap": 30000},  # plenty for 25K HC/C target
    "timschopf":    {"min_words":  50, "max_words": 500, "cap": None},
    "publichealth": {"min_words":  40, "max_words": 350, "cap": None},
}


def cmd_export(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled_keys, seen_hashes, _ = _load_state()

    sources = args.sources or list(SOURCE_CONFIG.keys())
    summary = {}

    for src in sources:
        if src not in SOURCES:
            print(f"  skip unknown source: {src}"); continue
        cfg = SOURCE_CONFIG[src]
        out_path = OUT_DIR / f"{src}.csv"
        written = 0
        dup_skip = 0
        already_labeled = 0
        local_seen = set()  # within this export

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLS)
            w.writeheader()
            for idx, text in SOURCES[src]():
                if (src, idx) in labeled_keys:
                    already_labeled += 1; continue
                wc = len(text.split())
                if not (cfg["min_words"] <= wc <= cfg["max_words"]):
                    continue
                h = _text_hash(text)
                if h in seen_hashes or h in local_seen:
                    dup_skip += 1; continue
                local_seen.add(h)
                w.writerow({
                    "source": src,
                    "source_idx": idx,
                    "text": text,
                    "word_count": wc,
                    "label": "",
                    "trigger": "",
                    "confidence": "",
                    "text_hash": h,
                })
                written += 1
                if cfg["cap"] and written >= cfg["cap"]:
                    break

        summary[src] = {"file": str(out_path), "written": written,
                        "dup_skipped": dup_skip, "already_labelled": already_labeled}
        print(f"  {src}: wrote {written} rows  -> {out_path}  "
              f"(dropped {dup_skip} dups, {already_labeled} already-labelled)")

    print()
    print("Total rows ready to label:", sum(s["written"] for s in summary.values()))


def cmd_ingest(args):
    """Ingest a filled CSV (subset of OUT_COLS, with label populated)."""
    path = Path(args.path)
    if not path.exists():
        sys.exit(f"Not found: {path}")

    labeled_keys, seen_hashes, _ = _load_state()
    out_rows, skipped_blank, skipped_dup, skipped_invalid = [], 0, 0, 0

    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            label = (row.get("label") or "").strip().upper()
            if not label:
                skipped_blank += 1; continue
            if label not in VALID_LABELS:
                skipped_invalid += 1; continue
            src, idx = row["source"], row["source_idx"]
            if (src, idx) in labeled_keys:
                skipped_dup += 1; continue
            h = row.get("text_hash") or _text_hash(row.get("text", ""))
            if h in seen_hashes:
                skipped_dup += 1; continue
            seen_hashes.add(h)
            out_rows.append({
                "source": src, "source_idx": idx, "text": row["text"],
                "label": label,
                "trigger": (row.get("trigger") or "").strip(),
                "confidence": (row.get("confidence") or "high").strip().lower(),
                "text_hash": h,
            })

    _append(out_rows)
    print(f"Appended {len(out_rows)} labelled rows from {path.name}")
    print(f"  Skipped: {skipped_blank} unlabelled, {skipped_dup} duplicates, "
          f"{skipped_invalid} invalid labels")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="produce labelling-ready CSVs")
    pe.add_argument("--sources", nargs="*", default=None,
                    help="subset of sources to export (default: all)")
    pe.set_defaults(func=cmd_export)

    pi = sub.add_parser("ingest", help="ingest a filled CSV back into medical_data_v2.csv")
    pi.add_argument("path", help="path to the filled CSV")
    pi.set_defaults(func=cmd_ingest)

    args = p.parse_args()
    args.func(args)
