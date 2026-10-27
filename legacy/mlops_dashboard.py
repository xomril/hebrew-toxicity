import os
import json
import datetime
import pandas as pd
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import evaluate

# הגדרת נתיבים
DATA_FILE = "data.json"
MODELS_DIR = "./model_registry"

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. ניהול דאטה (Data Management)
# ---------------------------------------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["text", "label"])
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def save_data(df):
    data = df.to_dict(orient="records")
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_example(text, label_val):
    if not text or not str(text).strip():
        return load_data(), "אנא הכנס טקסט תקין"
    
    df = load_data()
    new_row = pd.DataFrame([{"text": text, "label": int(label_val)}])
    df = pd.concat([df, new_row], ignore_index=True)
    
    save_data(df)
    total_count = len(df)
    return df, f"הנתון נוסף בהצלחה! סך הכל דוגמאות: {total_count}"

# ---------------------------------------------------------
# 2. אימון מודל
# ---------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}

def train_new_model(model_name_version, epochs, lr, batch_size):
    df = load_data()
    if len(df) < 10:
        return "יש להזין לפחות 10 דוגמאות לפני תחילת האימון"
    
    output_model_path = os.path.join(MODELS_DIR, model_name_version)
    
    hf_dataset = Dataset.from_pandas(df)
    ds_split = hf_dataset.train_test_split(test_size=0.2)
    
    BASE_MODEL = "dicta-il/dictabert"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    
    tokenized_ds = ds_split.map(tokenize_fn, batched=True)
    
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name_version}",
        eval_strategy="epoch",
        learning_rate=float(lr),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        num_train_epochs=int(epochs),
        weight_decay=0.01,
        use_mps_device=torch.backends.mps.is_available(),
        logging_steps=5,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    
    eval_results = trainer.evaluate()
    acc = round(eval_results.get("eval_accuracy", 0) * 100, 2)
    f1 = round(eval_results.get("eval_f1", 0) * 100, 2)
    
    return f"האימון הושלם!\nנשמר ב: {output_model_path}\nAccuracy: {acc}%\nF1 Score: {f1}%"

# ---------------------------------------------------------
# 3. ניתוח, בדיקה ופידבק (Inference & Active Learning)
# ---------------------------------------------------------
def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
    return models if models else []

def predict(selected_model_name, text):
    if not selected_model_name or selected_model_name == "אין מודלים מאומנים":
        return "בחר מודל תקין מהרשימה", None
    
    model_path = os.path.join(MODELS_DIR, selected_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    toxic_prob = probs[0][1].item() * 100
    safe_prob = probs[0][0].item() * 100
    
    predicted_label = 1 if toxic_prob > 50 else 0
    
    if predicted_label == 1:
        res = f"תגובה רעילה / עוקצנית ({toxic_prob:.1f}%)"
    else:
        res = f"תגובה חיובית / ניטרלית ({safe_prob:.1f}%)"
        
    return res, predicted_label

def process_feedback(text, predicted_label, is_correct):
    if not text or predicted_label is None:
        return "אין משפט פעיל לבדיקה"
    
    if is_correct:
        # המודל צדק - נוסיף את הדוגמה כמו שהיא
        target_label = predicted_label
        msg = "תודה! הניחוי הנכון נשמר ב-Dataset לחיזוק המודל."
    else:
        # המודל טעה - נהפוך את התגית (0 ל-1, או 1 ל-0)
        target_label = 1 if predicted_label == 0 else 0
        msg = f"תודה! התיקון נשמר ב-Dataset בתור label={target_label} לאימון הבא."
    
    add_example(text, target_label)
    return msg

# ---------------------------------------------------------
# 4. ממשק המשתמש (Gradio Dashboard)
# ---------------------------------------------------------
css_rtl = """
.gradio-container {
    direction: rtl !important;
    text-align: right !important;
    font-family: system-ui, -apple-system, sans-serif !important;
}
input, textarea, table, .table-wrap, td, th {
    direction: rtl !important;
    text-align: right !important;
}
"""

with gr.Blocks(title="Dicta MLOps Studio", css=css_rtl, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dicta MLOps Studio")
    
    # State לשמירת תוצאת הבדיקה הנוכחית
    current_text_state = gr.State("")
    current_pred_state = gr.State(None)
    
    with gr.Tabs():
        # לשונית 1: ניהול דאטה
        with gr.TabItem("ניהול דאטה-סט"):
            gr.Markdown("### הוספת דוגמאות וצפייה בנתונים")
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(label="טקסט להוספה", placeholder="הכנס תגובה...")
                    input_label = gr.Radio([("חיובי (0)", 0), ("רעיל (1)", 1)], label="תגית", value=0)
                    add_btn = gr.Button("הוסף ל-Dataset", variant="primary")
                    status_msg = gr.Textbox(label="סטטוס", interactive=False)
                with gr.Column(scale=3):
                    dataset_table = gr.Dataframe(value=load_data(), headers=["text", "label"], interactive=False, label="הדאטה הנוכחי")
            
            add_btn.click(add_example, inputs=[input_text, input_label], outputs=[dataset_table, status_msg])

        # לשונית 2: אימון מודל
        with gr.TabItem("אימון מודל"):
            gr.Markdown("### כוונון פרמטרים והרצת אימון")
            with gr.Row():
                with gr.Column():
                    version_input = gr.Textbox(label="שם גרסת המודל", value=f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
                    epochs_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Epochs")
                    lr_input = gr.Dropdown(choices=["1e-5", "2e-5", "5e-5", "1e-4"], value="2e-5", label="Learning Rate")
                    batch_size_input = gr.Dropdown(choices=["4", "8", "16", "32"], value="8", label="Batch Size")
                    train_btn = gr.Button("התחל אימון", variant="primary")
                with gr.Column():
                    train_output = gr.Textbox(label="תוצאות אימון", lines=10, interactive=False)
            
            train_btn.click(train_new_model, inputs=[version_input, epochs_slider, lr_input, batch_size_input], outputs=[train_output])

        # לשונית 3: בדיקת מודלים ופידבק (Active Learning)
        with gr.TabItem("בדיקה ופידבק (Active Learning)"):
            gr.Markdown("### בדיקת מודל ומשוב לשיפור ה-Dataset")
            with gr.Row():
                with gr.Column():
                    available_models = get_available_models()
                    init_val = available_models[0] if available_models else None
                    model_selector = gr.Dropdown(choices=available_models, label="בחר גרסת מודל", value=init_val)
                    refresh_models_btn = gr.Button("רענן רשימה")
                    test_text = gr.Textbox(label="טקסט לבדיקה", placeholder="הקלד משפט...")
                    predict_btn = gr.Button("בחינה", variant="primary")
                with gr.Column():
                    prediction_output = gr.Textbox(label="תוצאת המודל", lines=2, interactive=False)
                    
                    gr.Markdown("#### האם המודל צדק?")
                    with gr.Row():
                        correct_btn = gr.Button("👍 צודק", variant="secondary")
                        wrong_btn = gr.Button("👎 טעה (תייג כהפוך)", variant="stop")
                    
                    feedback_status = gr.Textbox(label="סטטוס פידבק", interactive=False)

            def refresh_dropdown():
                models = get_available_models()
                new_value = models[0] if models else None
                return gr.update(choices=models, value=new_value)

            def run_prediction(model_name, text):
                res_str, pred_label = predict(model_name, text)
                return res_str, text, pred_label

            refresh_models_btn.click(refresh_dropdown, outputs=[model_selector])
            
            predict_btn.click(
                run_prediction, 
                inputs=[model_selector, test_text], 
                outputs=[prediction_output, current_text_state, current_pred_state]
            )
            
            correct_btn.click(
                lambda txt, pred: process_feedback(txt, pred, is_correct=True),
                inputs=[current_text_state, current_pred_state],
                outputs=[feedback_status]
            )
            
            wrong_btn.click(
                lambda txt, pred: process_feedback(txt, pred, is_correct=False),
                inputs=[current_text_state, current_pred_state],
                outputs=[feedback_status]
            )

demo.launch()