"""ייבוא דאטה-סט מקובץ JSON (רשימת {text, label}) לטבלת labeled_dataset: python -m scripts.import_json data.json"""

import argparse
import json

from sqlalchemy import select

from db import LabeledExample, SessionLocal


def main():
    parser = argparse.ArgumentParser(description="Import a JSON dataset into labeled_dataset")
    parser.add_argument("path")
    parser.add_argument("--source", default="seed")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    with SessionLocal() as session:
        # דילוג על דוגמאות שכבר קיימות כדי שאפשר יהיה להריץ שוב בבטחה
        existing = set(session.execute(select(LabeledExample.text, LabeledExample.label)).all())
        new = [r for r in rows if (r["text"], int(r["label"])) not in existing]
        session.add_all(LabeledExample(text=r["text"], label=int(r["label"]), source=args.source) for r in new)
        session.commit()

    print(f"Imported {len(new)} examples ({len(rows) - len(new)} already present)")


if __name__ == "__main__":
    main()
