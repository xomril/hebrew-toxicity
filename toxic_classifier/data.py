import json

import pandas as pd

from .config import DATA_FILE


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=["text", "label"])
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def save_data(df: pd.DataFrame) -> None:
    data = df.to_dict(orient="records")
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_example(text: str, label: int) -> pd.DataFrame:
    """מוסיף דוגמה לדאטה-סט ומחזיר את הדאטה המעודכן. זורק ValueError על טקסט ריק."""
    if not text or not str(text).strip():
        raise ValueError("empty text")

    df = load_data()
    new_row = pd.DataFrame([{"text": text, "label": int(label)}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df)
    return df
