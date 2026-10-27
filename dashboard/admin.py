"""דשבורד ניהול (MLOps Studio): ניהול דאטה, אימון, בדיקה ופידבק. שכבת UI בלבד - כל הלוגיקה ב-toxic_classifier."""

import gradio as gr

from db import SessionLocal
from db.repository import add_labeled_example, labeled_dataframe, record_feedback
from toxic_classifier import TrainConfig, list_models, predict, train_model
from toxic_classifier.training import new_version_name


# ---------------------------------------------------------
# Handlers - מתרגמים בין ה-UI לליבת ה-ML ולמסד הנתונים
# ---------------------------------------------------------
def load_data():
    with SessionLocal() as session:
        return labeled_dataframe(session)


def on_add_example(text, label_val):
    with SessionLocal() as session:
        try:
            add_labeled_example(session, text, int(label_val))
        except ValueError:
            return labeled_dataframe(session), "אנא הכנס טקסט תקין"
        df = labeled_dataframe(session)
    return df, f"הנתון נוסף בהצלחה! סך הכל דוגמאות: {len(df)}"


def on_train(version, epochs, lr, batch_size):
    cfg = TrainConfig(version=version, epochs=int(epochs), learning_rate=float(lr), batch_size=int(batch_size))
    try:
        result = train_model(cfg, load_data())
    except ValueError:
        return "יש להזין לפחות 10 דוגמאות לפני תחילת האימון"
    return (
        f"האימון הושלם!\nנשמר ב: {result.model_path}\n"
        f"Accuracy: {result.accuracy * 100:.2f}%\nF1 Score: {result.f1 * 100:.2f}%"
    )


def on_predict(model_name, text):
    if not model_name:
        return "בחר מודל תקין מהרשימה", text, None, None
    pred = predict(model_name, text)
    if pred.is_toxic:
        res = f"תגובה רעילה / עוקצנית ({pred.toxic_prob * 100:.1f}%)"
    else:
        res = f"תגובה חיובית / ניטרלית ({pred.safe_prob * 100:.1f}%)"
    return res, text, pred.label, model_name


def on_feedback(text, predicted_label, model_name, is_correct):
    if not text or predicted_label is None:
        return "אין משפט פעיל לבדיקה"

    # המודל צדק - התגית נשארת. המודל טעה - נהפוך את התגית (0 ל-1, או 1 ל-0)
    target_label = predicted_label if is_correct else 1 - predicted_label
    with SessionLocal() as session:
        record_feedback(session, target_label, text=text, predicted_label=predicted_label, model_version=model_name)

    if is_correct:
        return "תודה! הניחוי הנכון נשמר ב-Dataset לחיזוק המודל."
    return f"תודה! התיקון נשמר ב-Dataset בתור label={target_label} לאימון הבא."


def refresh_dropdown():
    models = list_models()
    return gr.update(choices=models, value=models[0] if models else None)


# ---------------------------------------------------------
# ממשק המשתמש (Gradio Dashboard)
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
    current_model_state = gr.State(None)

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
                    dataset_table = gr.Dataframe(value=load_data, headers=["text", "label"], interactive=False, label="הדאטה הנוכחי")

            add_btn.click(on_add_example, inputs=[input_text, input_label], outputs=[dataset_table, status_msg])

        # לשונית 2: אימון מודל
        with gr.TabItem("אימון מודל"):
            gr.Markdown("### כוונון פרמטרים והרצת אימון")
            with gr.Row():
                with gr.Column():
                    version_input = gr.Textbox(label="שם גרסת המודל", value=new_version_name)
                    epochs_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Epochs")
                    lr_input = gr.Dropdown(choices=["1e-5", "2e-5", "5e-5", "1e-4"], value="2e-5", label="Learning Rate")
                    batch_size_input = gr.Dropdown(choices=["4", "8", "16", "32"], value="8", label="Batch Size")
                    train_btn = gr.Button("התחל אימון", variant="primary")
                with gr.Column():
                    train_output = gr.Textbox(label="תוצאות אימון", lines=10, interactive=False)

            train_btn.click(on_train, inputs=[version_input, epochs_slider, lr_input, batch_size_input], outputs=[train_output])

        # לשונית 3: בדיקת מודלים ופידבק (Active Learning)
        with gr.TabItem("בדיקה ופידבק (Active Learning)"):
            gr.Markdown("### בדיקת מודל ומשוב לשיפור ה-Dataset")
            with gr.Row():
                with gr.Column():
                    available_models = list_models()
                    model_selector = gr.Dropdown(
                        choices=available_models,
                        label="בחר גרסת מודל",
                        value=available_models[0] if available_models else None,
                    )
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

            refresh_models_btn.click(refresh_dropdown, outputs=[model_selector])

            predict_btn.click(
                on_predict,
                inputs=[model_selector, test_text],
                outputs=[prediction_output, current_text_state, current_pred_state, current_model_state],
            )

            correct_btn.click(
                lambda txt, pred, model: on_feedback(txt, pred, model, is_correct=True),
                inputs=[current_text_state, current_pred_state, current_model_state],
                outputs=[feedback_status],
            )

            wrong_btn.click(
                lambda txt, pred, model: on_feedback(txt, pred, model, is_correct=False),
                inputs=[current_text_state, current_pred_state, current_model_state],
                outputs=[feedback_status],
            )

if __name__ == "__main__":
    demo.launch()
