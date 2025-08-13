import gradio as gr
import requests
import uuid

API_URL = "http://localhost:8000/chat"


def chat_with_bot(message, history, session_id):
    if not session_id:
        session_id = str(uuid.uuid4())
    payload = {"session_id": session_id, "text": message}
    response = requests.post(API_URL, json=payload)
    data = response.json()
    bot_reply = data.get("response", "عذراً، لم أتلقَّ رد.")
    history = history or []
    # Append new messages as dicts with role/content
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_reply})
    return history, session_id


def create_gradio_interface():
    with gr.Blocks() as demo:
        session_state = gr.State()
        chatbot = gr.Chatbot(
            label="مساعد حجز تذاكر كرة القدم الذكي (العربية)", type="messages")
        msg = gr.Textbox(label="اكتب رسالتك هنا...",
                         placeholder="اسألني عن المباريات أو احجز تذكرة...")
        session_id = gr.Textbox(value="", visible=False)

        def submit(message, history, session_id_value):
            return chat_with_bot(message, history, session_id_value)

        msg.submit(submit, inputs=[msg, chatbot, session_id], outputs=[
                   chatbot, session_id])
        return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
