import streamlit as st
import os
import time
from groq import Groq
from huggingface_hub import InferenceClient
from gtts import gTTS
import base64
from streamlit_mic_recorder import mic_recorder

from secret_api_keys import huggingface_api_key, groq_api_key

# ======================================================
# FFmpeg PATH for Whisper (Windows Fix)
# ======================================================


# ======================================================
# Center Popup Function
# ======================================================
def show_center_message(text, color="rgba(255,255,255,0.1)", duration=2):
    popup = st.empty()
    popup.markdown(
        f"""
        <div class="popup" style="border-color: {color}; box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 0 20px {color};">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(duration)
    popup.empty()

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ======================================================
# Load CSS Based on Theme
# ======================================================
def load_css(theme):
    css_file = "style.css" if theme == "dark" else "light_style.css"
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(st.session_state.theme)

# ======================================================
# LLM Models (DeepSeek / LLaMA / Groq)
# ======================================================
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

deepseek_client = InferenceClient("deepseek-ai/DeepSeek-V3.2-Exp", token=huggingface_api_key)
hf_llama_client = InferenceClient("meta-llama/Meta-Llama-3.1-8B-Instruct", token=huggingface_api_key)
groq_client = Groq(api_key=groq_api_key)

# ======================================================
# Language Options
# ======================================================
LANG_OPTIONS = {
    "English": {"code": "en", "name": "English"},
    "Hindi": {"code": "hi", "name": "Hindi"},
    "Gujarati": {"code": "gu", "name": "Gujarati"},
}

def apply_language_prompt(base_prompt, lang_name):
    return (
        base_prompt +
        f"\n\nIMPORTANT:\n"
        f"- Write the entire output in {lang_name}.\n"
        f"- Use the native script (Devanagari for Hindi, Gujarati script for Gujarati).\n"
        f"- Do NOT switch languages."
    )

# ======================================================
# Generate Text (LLM)
# ======================================================
def generate_with(model_name, prompt, lang_name):

    final_prompt = apply_language_prompt(prompt, lang_name)

    try:
        if model_name == "deepseek":
            resp = deepseek_client.chat_completion(
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=500
            )
            return resp.choices[0].message["content"]

        elif model_name == "hf_llama":
            resp = hf_llama_client.chat_completion(
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=500
            )
            return resp.choices[0].message["content"]

        elif model_name == "groq_llama":
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": final_prompt}]
            )
            return resp.choices[0].message.content

        return "❌ Unknown model"
    except Exception as e:
        error_msg = f"❌ Error in {model_name}: {str(e)}"
        st.error(error_msg)
        return None


# ======================================================
# Speech to Text (Groq Whisper API)
# ======================================================
def speech_to_text(audio_path, lang_code):
    try:
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                language=lang_code
            )
        return transcription.text.strip()
    except Exception as e:
        st.error(f"STT Error: {e}")
        return ""

# ======================================================
# Text to Speech (gTTS)
# ======================================================
def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_file = "tts_output.mp3"
        tts.save(audio_file)

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        b64 = base64.b64encode(audio_bytes).decode()

        audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        return audio_html

    except Exception as e:
        return f"TTS Error: {e}"

# ======================================================
# STREAMLIT UI
# ======================================================

# Render Logo at the top
if os.path.exists("logo.png"):
    col_logo_left, col_logo_center, col_logo_right = st.columns([1.5, 2, 1.5])
    with col_logo_center:
        st.image("logo.png", use_container_width=True)

col_title, col_theme = st.columns([8, 2])
with col_title:
    st.title("🤖 Blog Writing Assistant")
with col_theme:
    is_dark = st.session_state.theme == "dark"
    toggled = st.toggle("🌙 Dark Mode" if is_dark else "🌞 Light Mode", value=is_dark, key="theme_switch")
    new_theme = "dark" if toggled else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

top_col1, top_col2, top_col3 = st.columns([3, 2, 2])

with top_col2:
    model_choice = st.selectbox("Model", ["groq_llama", "hf_llama", "deepseek"])

with top_col3:
    selected_lang = st.selectbox("Language", list(LANG_OPTIONS.keys()))
    lang_code = LANG_OPTIONS[selected_lang]["code"]
    lang_name = LANG_OPTIONS[selected_lang]["name"]

# -------------------- Generate Titles --------------------
st.header("Generate Blog Titles")

if "voice_topic" not in st.session_state:
    st.session_state.voice_topic = ""

topic = st.text_input("Enter Topic:", value=st.session_state.voice_topic)

col_btn, col_mic = st.columns([4, 1])

with col_btn:
    gen_titles_btn = st.button("Generate Titles")

with col_mic:
    audio_data = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="🛑 Stop",
        key="mic_recorder_btn",
        use_voicerecorder=False
    )

if audio_data:
    audio_bytes = audio_data["bytes"]
    audio_file = "input.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)
    
    with st.spinner("Transcribing..."):
        text = speech_to_text(audio_file, lang_code)
    
    if text:
        st.session_state.voice_topic = text
        show_center_message(f"✔ Recognized: {text}", "rgba(76, 175, 80, 0.4)", 2)
        st.rerun()

# Generate Titles
if gen_titles_btn:
    with st.spinner("Generating Titles..."):
        prompt = (
            f"Generate exactly 10 blog titles about: \"{topic}\".\n"
            f"- Only output numbered titles.\n"
            f"- Stay strictly on topic."
        )
        result = generate_with(model_choice, prompt, lang_name)
        if result:
            st.session_state["titles"] = result
            st.session_state["titles_list"] = [t.strip() for t in result.split('\n') if t.strip()]

# Show Titles + TTS
if st.session_state.get("titles"):
    st.subheader("Generated Titles")
    
    if "titles_list" in st.session_state:
        st.markdown("<p style='color: #e2e8f0; margin-bottom: 15px;'>Click a title below to select it for your outline:</p>", unsafe_allow_html=True)
        for i, t in enumerate(st.session_state["titles_list"]):
            if st.button(t, key=f"title_btn_{i}"):
                clean_t = t.lstrip("0123456789.-* ")
                st.session_state.selected_title_input = clean_t
    else:
        st.write(st.session_state["titles"])

    if st.button("🔊 Speak Titles"):
        audio_html = text_to_speech(st.session_state["titles"], lang_code)
        st.markdown(audio_html, unsafe_allow_html=True)

# -------------------- Outline + Blog --------------------
st.header("Generate Outline")

if "selected_title_input" not in st.session_state:
    st.session_state.selected_title_input = ""

selected_title = st.text_input("Enter selected title:", key="selected_title_input")

colA, colB = st.columns(2)

with colA:
    if st.button("Generate Outline"):
        with st.spinner("Generating Outline..."):
            outline_prompt = (
                f"Create a detailed outline for: \"{selected_title}\".\n"
                f"- Use numbered sections.\n"
                f"- Use bullet points."
            )
            st.session_state["outline"] = generate_with(model_choice, outline_prompt, lang_name)
            st.session_state["full_blog"] = None

with colB:
    if st.button("Generate Full Blog"):
        with st.spinner("Generating Full Blog..."):
            outline = st.session_state.get("outline", "")
            blog_prompt = f"""
            Write a full 1200-1600 word blog using this outline:

            {outline}

            Keep language simple and structured.
            """
            st.session_state["full_blog"] = generate_with(model_choice, blog_prompt, lang_name)
            st.session_state["outline"] = None

# Show Outline + TTS
if st.session_state.get("outline"):
    st.subheader("Generated Outline")
    st.write(st.session_state["outline"])

    if st.button("🔊 Speak Outline"):
        audio_html = text_to_speech(st.session_state["outline"], lang_code)
        st.markdown(audio_html, unsafe_allow_html=True)

# Show Blog + TTS
elif st.session_state.get("full_blog"):
    st.subheader("Generated Blog")
    st.markdown(st.session_state["full_blog"], unsafe_allow_html=True)

    if st.button("🔊 Speak Blog"):
        audio_html = text_to_speech(st.session_state["full_blog"], lang_code)
        st.markdown(audio_html, unsafe_allow_html=True)

