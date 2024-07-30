import gradio as gr
from transformers import pipeline

def chatbot_response(text):
    response = chat(text)
    return response

# Load pre-trained model
chat = pipeline("text-generation", model="gpt-2", max_length=100)

# Create Gradio interface
interface = gr.Interface(fn=chatbot_response,
                         inputs=gr.inputs.Textbox(lines=2, placeholder="Type your question here..."),
                         outputs="text",
                         title="ChatGPT Chatbot",
                         description="This is a demo for shiwen")

if __name__ == "__main__":
    interface.launch()
