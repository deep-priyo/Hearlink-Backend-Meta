# Media processing imports
import os
import re
import tempfile
from pathlib import Path

# Document processing imports
import PyPDF2
import docx
import google.generativeai as genai
import requests
import whisper
import yt_dlp
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
# Translation and language processing imports
from langdetect import detect
from moviepy import VideoFileClip
# AI and ML imports
from openai import OpenAI

# Environment and configuration
import uuid
import os
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS
import yt_dlp
import whisper
from pathlib import Path
import tempfile
# Import statements
import uuid
import os
import subprocess

from dotenv import load_dotenv
from faster_whisper import WhisperModel
import openai

# Load environment variables
load_dotenv()

# AI API configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# OpenAI client for LLaMA
client = OpenAI(
    base_url="https://infer.e2enetworks.net/project/p-5454/genai/llama_3_3_70b_instruct_fp8/v1",
    api_key=os.getenv("E2E_API_KEY")
)


# =============================================================================
# MEDIA PROCESSING FUNCTIONS
# =============================================================================

def extract_audio(video_path):
    """Extracts audio from a video file."""
    audio_path = video_path.replace(".mp4", ".wav")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")
    return audio_path


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text


def extract_text_from_docx(file_path):
    """Extract text from a Word (.docx) file."""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


# =============================================================================
# LANGUAGE PROCESSING AND TRANSLATION FUNCTIONS
# =============================================================================

def detect_language(text):
    """Detects the language of the given text using langdetect."""
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails


def translate_text(text, target_language):
    """Translates the transcribed text into the selected language."""
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"


def translate_notes(notes_text, target_language):
    """Translate notes to target language using Gemini."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Translate the following text into {target_language}:\n\n{notes_text}")
    return response.text if response else notes_text


# =============================================================================
# YOUTUBE TRANSCRIPT PROCESSING FUNCTIONS
# =============================================================================

transcription_store = {}

# Language mapping
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu"
}


# Load Whisper model (global variable for reuse)
def load_model():
    # try:
    #     # Try GPU first
    #     return WhisperModel("large-v3", device="cuda", compute_type="float16")
    # except Exception as e:
    #     print(f"GPU not available, falling back to CPU: {str(e)}")
    #     # Fallback to CPU with a smaller model for better performance
    #     # Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
    return WhisperModel("base", device="cpu", compute_type="int8")


model = load_model()


# Helper functions
def download_audio(video_url):
    """Download YouTube audio using yt-dlp."""
    unique_name = f"audio_{uuid.uuid4().hex}.mp3"
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "--output", unique_name,
        video_url
    ]
    try:
        subprocess.run(command, check=True)
        return unique_name
    except Exception as e:
        print(f"Audio download error: {str(e)}")
        return None


def transcribe_audio_faster_whisper(file_path):
    """Transcribe using Whisper with auto language detection."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
            print("Audio file is empty or corrupt.")
            return None, None

        print("Transcribing with Whisper (auto language)...")
        segments, info = model.transcribe(file_path, beam_size=2)
        transcript = " ".join([seg.text.strip() for seg in segments])

        if not transcript.strip():
            print("Whisper returned an empty transcript.")
            return None, info.language

        return transcript, info.language
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None, None

def translate_transcript(transcript_text, target_lang_code):
    """Translate transcript using LLaMA or GPT."""
    try:
        target_language = LANG_MAP.get(target_lang_code, "English")
        prompt = (
            f"Translate the following transcript to {target_language}.\n\n"
            f"{transcript_text}"
        )

        response = openai.chat.completions.create(
            model="llama_3_1_405b_instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
            stream=False
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Translation error: {str(e)}")
        return transcript_text

def generate_detailed_notes(transcript_text, target_lang_code):
    """Generate detailed notes using LLaMA."""
    try:
        language = LANG_MAP.get(target_lang_code, "English")
        prompt = (
            f"You are an expert student note summarizer. Summarize the transcript into clear, structured notes in {language} only.\n"
            f"⚠️ Do not use English. Only respond in {language}.\n"
            f"Use bullet points, subheadings, and numbered lists.\n\n"
            f"Transcript:\n{transcript_text}"
        )

        response = openai.chat.completions.create(
            model="llama_3_1_405b_instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
            stream=False
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Note generation error: {str(e)}")
        return transcript_text


# =============================================================================
# AI CONTENT GENERATION FUNCTIONS
# =============================================================================

def call_e2e_llama4(prompt):
    """Send prompt to E2E LLaMA 3.3 API and return the output."""
    try:
        response = client.chat.completions.create(
            model='llama_3_3_70b_instruct_fp8',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None


def summarize_text(text, lang):
    """Summarizes the text and extracts key points in bullet format."""
    language_names = {
        "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
        "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"
    }

    try:
        lang_name = language_names.get(lang, lang)
        prompt = f"""You are a multilingual assistant.

    IMPORTANT: Your response MUST be in {lang_name} language ({lang}).
    - DO NOT respond in English. Respond ONLY in {lang_name}.
    - Extract only the most important points from the following text.
    - Use clear and concise bullet points.
    - Format your response in a structured, easy-to-read format.

    Text to summarize:
    """

        completion = client.chat.completions.create(
            model="llama_3_3_70b_instruct_fp8",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1,
            stream=True
        )

        summary = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                summary += chunk.choices[0].delta.content

        return summary
    except Exception as e:
        print(f"❌ API Error: {e}")
        return "Error: Failed to summarize due to API issue."


def generate_flashcards(summary_text):
    """Formats the summary into structured flashcards."""
    flashcards = []
    summary_text = str(summary_text)
    points = [point.strip() for point in summary_text.split("\n") if point.strip()]

    for idx, point in enumerate(points, start=1):
        flashcards.append(f"📌 **Key Point {idx}:** {point}")

    return "\n".join(flashcards)


def generate_quiz(text):
    """Generate a structured quiz from the input text using LLaMA 3.3 model."""
    prompt = f"""
    You are an educational assistant. Create a multiple-choice quiz from the text.
    Provide exactly 5 questions, each with 4 options (A, B, C, D), and mark the correct answer separately.
    Keep the language the same as the input text.

    Format the output clearly like:
    1. Question text?
       Option 1
       Option 2
       Option 3
       Option 4

    After listing all 5 questions, provide the correct answers separately in this format:

    **Correct Answers:**
    1. A
    2. C
    3. B
    4. D
    5. A

    Text:
    {text}
    """

    response_text = call_e2e_llama4(prompt)
    if not response_text:
        return {"error": "No response from LLaMA model"}

    quiz_questions = []
    correct_answers = {}

    try:
        questions_part, answers_part = response_text.split("**Correct Answers:**")

        questions_blocks = questions_part.strip().split("\n\n")
        for block in questions_blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 5:
                question = lines[0].strip().replace("?", "")
                options = [line.strip() for line in lines[1:5]]

                quiz_questions.append({
                    "question": question,
                    "options": options,
                })

        for line in answers_part.strip().split("\n"):
            if "." in line:
                number, ans = line.strip().split(".")
                correct_answers[int(number.strip())] = ans.strip()

        structured_quiz = []
        for i, q in enumerate(quiz_questions):
            structured_quiz.append({
                "question": q["question"],
                "options": q["options"],
                "answer": correct_answers.get(i + 1)
            })

        return structured_quiz

    except Exception as e:
        return {"error": f"Failed to parse quiz: {str(e)}"}


def generate_exercises(text):
    """Generate structured exercises without JSON."""
    prompt = """
    You are an educational assistant. Create structured exercises from the text.
    - 5 Fill-in-the-blank questions (missing words marked as '_____')
    - 5 Short-answer questions (1-2 sentence responses)
    - 5 Long-answer questions (detailed responses)

    Format the output clearly like:

    **Fill in the Blanks**
    1. Sentence with _____ missing.

    **Short Answer Questions**
    1. What is the importance of X?

    **Long Answer Questions**
    1. Explain how X impacts Y in detail.

    After listing all questions, provide the correct answers separately in this format:

    **Answers:**

    **Fill in the Blanks**
    1. Correct answer
    2. Correct answer

    **Short Answer Questions**
    1. Answer

    **Long Answer Questions**
    1. Answer

    Text:
    """ + text

    return call_e2e_llama4(prompt)


def parse_exercise_response(text):
    """Parse exercise response into structured format."""
    try:
        fill_blanks = re.findall(r'\*\*Fill in the Blanks\*\*\s*(.*?)\*\*Short Answer Questions\*\*', text, re.S)[
            0].strip()
        short_answers = re.findall(r'\*\*Short Answer Questions\*\*\s*(.*?)\*\*Long Answer Questions\*\*', text, re.S)[
            0].strip()
        long_answers = re.findall(r'\*\*Long Answer Questions\*\*\s*(.*?)\*\*Answers:\*\*', text, re.S)[0].strip()
        answers_section = re.findall(r'\*\*Answers:\*\*\s*(.*)', text, re.S)[0].strip()

        fb_answers = \
            re.findall(r'\*\*Fill in the Blanks\*\*\s*(.*?)\*\*Short Answer Questions\*\*', answers_section, re.S)[
                0].strip()
        sa_answers = \
            re.findall(r'\*\*Short Answer Questions\*\*\s*(.*?)\*\*Long Answer Questions\*\*', answers_section, re.S)[
                0].strip()
        la_answers = re.findall(r'\*\*Long Answer Questions\*\*\s*(.*)', answers_section, re.S)[0].strip()

        def extract_items(block):
            return [re.sub(r'^\d+\.\s*', '', line.strip()) for line in block.strip().split('\n') if line.strip()]

        return {
            "fillBlanks": extract_items(fill_blanks),
            "shortAnswer": extract_items(short_answers),
            "longAnswer": extract_items(long_answers),
            "answers": {
                "fillBlanks": extract_items(fb_answers),
                "shortAnswer": extract_items(sa_answers),
                "longAnswer": extract_items(la_answers)
            }
        }

    except Exception as e:
        print("Parsing error:", str(e))
        return None


def generate_detailed_notes(transcript_text, language):
    """Generate detailed notes from transcript."""
    prompt = f"You are a YouTube video summarizer. Summarize the transcript into key points and detailed notes in {language}."
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"{prompt}\n\n{transcript_text}")
    return response.text if response else transcript_text


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Transcript file determination
translated_file = "translated_transcript.txt"


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_transcript():
    """Read the translated transcript."""
    if os.path.exists(translated_file):
        with open(translated_file, "r", encoding="utf-8") as file:
            return file.read()
    else:
        return None


def get_latest_transcript():
    """Determine which transcript file to use."""
    if os.path.exists("translated_transcript.txt") and os.path.getsize("translated_transcript.txt") > 0:
        return "translated_transcript.txt"
    elif os.path.exists("transcript.txt") and os.path.getsize("transcript.txt") > 0:
        return "transcript.txt"
    else:
        raise FileNotFoundError("No valid transcript found. Run transcription first.")


def save_text_to_file(text, filename):
    """Save text to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
