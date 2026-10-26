"""
Build the processed data files used by the tools.

    python -m data_pipeline.build

Writes:
  data/processed/opportunities.json   (Kaggle data if present, else synthetic)
  data/processed/students.json        (synthetic + optional Google Form import)
  data/processed/meta.json            (which opportunity source was used)
"""

from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.generate_students import generate_students
from data_pipeline.import_form import import_form
from data_pipeline.load_kaggle import load_opportunities

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def build_all(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, verbose: bool = True) -> dict:
    """Run the whole pipeline and write JSON files. Returns a small summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    opportunities, source = load_opportunities(raw_dir)

    students = generate_students()
    # Optional drop-in: a Google Form export with "form" in its file name.
    for form_csv in sorted(Path(raw_dir).glob("*form*.csv")):
        students += import_form(form_csv)

    _write_json(out_dir / "opportunities.json", [o.to_dict() for o in opportunities])
    _write_json(out_dir / "students.json", [s.to_dict() for s in students])
    summary = {
        "opportunity_source": source,
        "n_opportunities": len(opportunities),
        "n_students": len(students),
    }
    _write_json(out_dir / "meta.json", summary)

    if verbose:
        print(f"[build] {len(opportunities)} opportunities (source: {source}), "
              f"{len(students)} students -> {out_dir}")
        if source == "synthetic":
            print("[build] Tip: put the Kaggle Internshala CSV in data/raw/ "
                  "(any file name containing 'internshala') to use real listings.")
    return summary


if __name__ == "__main__":
    build_all()
