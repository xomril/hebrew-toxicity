import gradio as gr
from transformers import pipeline

# טעינת המודל שאומן ונשמר מקומית
classifier = pipeline("text-classification", model="./my_hebrew_toxic_model", tokenizer="./my_hebrew_toxic_model")

def analyze_text(text):
    result = classifier(text)[0]
    label = "תרגיע, זו תגובה רעילה/עוקצנית! 🤬" if result['label'] == 'LABEL_1' else "תגובה סבבה לגמרי 👍"
    score = round(result['score'] * 100, 2)
    return f"{label} (ביטחון המודל: {score}%)"

# יצירת ממשק Gradio
demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=2, placeholder="הכנס תגובה בעברית לבדיקה..."),
    outputs="text",
    title="Dicta Toxic Comment Detector",
    description="מודל Dicta שעבר Fine-Tuning מקומי על ה-Mac שלי"
)

demo.launch()