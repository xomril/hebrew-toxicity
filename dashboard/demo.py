"""ממשק דמו ציבורי: בדיקת תגובה מול המודל העדכני ברג'יסטרי."""

import os

import gradio as gr

from toxic_classifier import latest_model, predict

# ניתן לקבע גרסה דרך משתנה סביבה, אחרת - הגרסה החדשה ביותר
MODEL_VERSION = os.environ.get("TOXIC_MODEL_VERSION") or latest_model()


def analyze_text(text):
    if not MODEL_VERSION:
        return "אין מודל מאומן ברג'יסטרי"
    pred = predict(MODEL_VERSION, text)
    label = "תרגיע, זו תגובה רעילה/עוקצנית! 🤬" if pred.is_toxic else "תגובה סבבה לגמרי 👍"
    return f"{label} (ביטחון המודל: {pred.confidence * 100:.2f}%)"


demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=2, placeholder="הכנס תגובה בעברית לבדיקה..."),
    outputs="text",
    title="Dicta Toxic Comment Detector",
    description=f"מודל Dicta שעבר Fine-Tuning מקומי (גרסה: {MODEL_VERSION})",
)

if __name__ == "__main__":
    demo.launch()
