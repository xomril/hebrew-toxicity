"""אימון מודל משורת הפקודה על הדאטה-סט שבמסד הנתונים: python -m scripts.train --epochs 5"""

import argparse

from db import SessionLocal
from db.repository import labeled_dataframe
from toxic_classifier.training import TrainConfig, train_model


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a Hebrew toxic-comment classifier")
    parser.add_argument("--version", default=defaults.version)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    args = parser.parse_args()

    with SessionLocal() as session:
        df = labeled_dataframe(session)

    result = train_model(
        TrainConfig(version=args.version, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size),
        df,
    )
    print(f"Saved {result.version} to {result.model_path}")
    print(f"Accuracy: {result.accuracy:.2%}  F1: {result.f1:.2%}")


if __name__ == "__main__":
    main()
