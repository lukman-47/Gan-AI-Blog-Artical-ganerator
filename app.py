import streamlit as st
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import torch
torch._dynamo.disable()

import whisper
from groq import Groq
from huggingface_hub import InferenceClient
from secret_api_keys import huggingface_api_key, groq_api_key

# ======================================================
# FIX: FFmpeg Path (Required for Whisper on Windows)
# ======================================================
FFMPEG_PATH = r"C:\Users\Lukman\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# ======================================================
# Load Whisper Model (Multilingual)
# ======================================================
stt_model = whisper.load_model("tiny")   # "small" = better but slower

# ======================================================
# Helper: Center Popup Message
# ======================================================
def show_center_message(text, color="#4B7BE5", duration=2):
    popup = st.empty()
    popup.markdown(
        f"""
        <style>
            .popup {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 18px 26px;
                border-radius: 10px;
                border: 2px solid {color};
                font-size: 20px;
                font-weight: 600;
                text-align: center;
                z-index: 99999;
            }}
        </style>
        <div class="popup">{text}</div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(duration)
    popup.empty()

# ======================================================
# Load CSS
# ======================================================
def load_css():
    if os.path.exists("style.css"):
        with open("style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ======================================================
# Text Generation Models (DeepSeek, LLaMA, Groq)
# ======================================================
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

deepseek_client = InferenceClient("deepseek-ai/DeepSeek-V3.2-Exp", token=huggingface_api_key)
hf_llama_client = InferenceClient("meta-llama/Meta-Llama-3.1-8B-Instruct", token=huggingface_api_key)
groq_client = Groq(api_key=groq_api_key)

# ======================================================
# Multilingual Settings
# ======================================================
LANG_OPTIONS = {
    "English": {"code": "en", "name": "English"},
    "Hindi":   {"code": "hi", "name": "Hindi"},
    "Gujarati": {"code": "gu", "name": "Gujarati"},
}

def apply_language_instruction(base_prompt: str, lang_name: str) -> str:
    """
    Add an instruction so the model answers in the chosen language + script.
    """
    return (
        base_prompt
        + f"\n\nIMPORTANT:\n"
          f"- Write the entire answer in {lang_name}.\n"
          f"- Use the native writing script of {lang_name} "
          f"(for example, Devanagari for Hindi, Gujarati script for Gujarati).\n"
          f"- Do NOT use any other language."
    )

# ======================================================
# TEXT GENERATION HANDLER
# ======================================================
def generate_with(model_name, prompt, lang_name="English"):
    final_prompt = apply_language_instruction(prompt, lang_name)

    if model_name == "deepseek":
        resp = deepseek_client.chat_completion(
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=600
        )
        return resp.choices[0].message["content"]

    elif model_name == "hf_llama":
        resp = hf_llama_client.chat_completion(
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=600
        )
        return resp.choices[0].message["content"]

    elif model_name == "groq_llama":
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return resp.choices[0].message.content

    return "❌ Unknown model"

# ======================================================
# RECORD AUDIO
# ======================================================
def record_audio(duration=5, filename="input.wav"):
    sample_rate = 16000
    st.write("🎙 Recording...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    audio_int16 = (audio * 32767).astype(np.int16)
    write(filename, sample_rate, audio_int16)

    return filename

# ======================================================
# SPEECH TO TEXT (Whisper, Multilingual)
# ======================================================
def speech_to_text(audio_path: str, lang_code: str):
    """
    lang_code: "en", "hi", "gu"
    """
    try:
        # Whisper supports manual language hint
        result = stt_model.transcribe(
            audio_path,
            language=lang_code,   # hint language
            task="transcribe"
        )
        return result["text"].strip()
    except Exception as e:
        st.error(f"STT Error: {e}")
        return ""

# ======================================================
# STREAMLIT UI
# ======================================================

st.title("📘 AI Blog Writing Assistant")

top_col1, top_col2, top_col3 = st.columns([3, 2, 2])

with top_col1:
    st.markdown("###")

with top_col2:
    model_choice = st.selectbox("Model", ["groq_llama", "hf_llama", "deepseek"])

with top_col3:
    lang_label = st.selectbox("Language", list(LANG_OPTIONS.keys()))
    lang_code = LANG_OPTIONS[lang_label]["code"]
    lang_name = LANG_OPTIONS[lang_label]["name"]

# -------------------- Generate Titles --------------------
st.header("Generate Blog Titles")

if "voice_topic" not in st.session_state:
    st.session_state.voice_topic = ""

topic = st.text_input("Enter Topic:", value=st.session_state.voice_topic)

col_btn, col_mic = st.columns([4, 1])

with col_btn:
    gen_titles_btn = st.button("Generate Titles")

with col_mic:
    if st.button("🎤", help="Use microphone"):
        show_center_message("🎙 Speak now...", "#FFC107", 2)
        audio_file = record_audio(5)
        text = speech_to_text(audio_file, lang_code=lang_code)

        if text:
            st.session_state.voice_topic = text
            show_center_message(f"✔ Recognized: {text}", "#4CAF50", 2)
            st.rerun()
        else:
            show_center_message("❌ Could not understand.", "#E53935", 2)

# Generate titles
if gen_titles_btn:
    base_prompt = (
        f"Generate exactly 10 blog titles about: \"{topic}\".\n"
        f"- Only output a numbered list of titles.\n"
        f"- No introduction or explanation.\n"
        f"- Stay strictly on the topic."
    )
    titles = generate_with(model_choice, base_prompt, lang_name=lang_name)
    st.write(titles)

# -------------------- Generate Outline & Blog --------------------
st.header("Generate Outline")

selected_title = st.text_input("Enter selected title:")

colA, colB = st.columns(2)

with colA:
    if st.button("Generate Outline"):
        outline_prompt = (
            f"Create a detailed blog outline for the title: \"{selected_title}\".\n"
            f"- Use numbered headings.\n"
            f"- Use nested bullet points where needed.\n"
            f"- Do not add any introduction text before the outline."
        )
        st.session_state["outline"] = generate_with(
            model_choice, outline_prompt, lang_name=lang_name
        )
        st.session_state["full_blog"] = None

with colB:
    if st.button("Generate Full Blog"):
        outline = st.session_state.get("outline", "")
        blog_prompt = f"""
        You are a professional blog writer.

        TITLE:
        {selected_title}

        OUTLINE (if provided, follow it closely):
        {outline}

        Write a complete blog article based on the title and outline.

        Requirements:
        - Length: around 1200–1600 words.
        - Use clear headings and subheadings.
        - Keep language simple and engaging.
        - Maintain smooth flow between sections.
        """
        st.session_state["full_blog"] = generate_with(
            model_choice, blog_prompt, lang_name=lang_name
        )
        st.session_state["outline"] = None

# -------------------- Output Section --------------------
if st.session_state.get("outline"):
    st.subheader("Generated Outline")
    st.write(st.session_state["outline"])

elif st.session_state.get("full_blog"):
    st.subheader("Generated Blog")
    st.markdown(st.session_state["full_blog"], unsafe_allow_html=True)
