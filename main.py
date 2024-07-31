import gradio as gr
from similar_search import similar_search

def greet(query,intensity):
    return similar_search(query) + intensity*"!"

demo = gr.Interface(
    fn=greet,
    inputs=["text","slider"],
    outputs=["text"]
)

demo.launch(share=True)
