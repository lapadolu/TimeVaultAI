import os
import gradio as gr
import tempfile
import ffmpeg
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

def extract_video_frame(video_path):
    out_path = tempfile.mktemp(suffix=".jpg")
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    midpoint = duration / 2
    (
        ffmpeg
        .input(video_path, ss=midpoint)
        .output(out_path, vframes=1)
        .run(capture_stdout=True, capture_stderr=True)
    )
    return out_path

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def generate_narrative(text_input, video_file, image_file):
    video_frame_path = extract_video_frame(video_file.name) if video_file else None
    video_frame_b64 = image_to_base64(video_frame_path) if video_frame_path else ""
    image_b64 = image_to_base64(image_file.name) if image_file else ""

    prompt = f"""
    You are an AI journal assistant. Today, the user uploaded the following:

    Text note: "{text_input}"
    Photo: ![user photo](data:image/jpeg;base64,{image_b64})
    Video frame: ![video frame](data:image/jpeg;base64,{video_frame_b64})

    Please create a deeply reflective, emotional yet concise narrative in the first person, as if written by the user themselves. Use literary language but remain authentic.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an emotionally intelligent AI diarist."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

iface = gr.Interface(
    fn=generate_narrative,
    inputs=[
        gr.Textbox(label="Scrivi brevemente cosa hai fatto oggi"),
        gr.File(label="Carica un video della tua giornata (mp4)"),
        gr.File(label="Carica una foto significativa (jpg/png)")
    ],
    outputs=gr.Textbox(label="Narrativa Generata"),
    title="NarrAI - Il tuo diario AI",
    description="Carica contenuti della tua giornata e ottieni una narrativa personale generata dall'intelligenza artificiale."
)

iface.launch()
