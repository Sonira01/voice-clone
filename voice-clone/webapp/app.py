import gradio as gr
from scripts.inference import generate_voice

gr.Interface(fn=generate_voice, inputs="text", outputs="audio").launch()
