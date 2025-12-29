import os
import time
import sqlite3
import speech_recognition as sr
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from fpdf import FPDF
import openai
import tempfile
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

# Set up custom LLaMA 3 API
openai.api_key = os.getenv("E2E_API_KEY")
openai.base_url = "https://infer.e2enetworks.net/project/p-5518/genai/llama_3_1_405b_instruct/v1/"


def call_llm(prompt, context=""):
    """Function to stream LLM response"""
    try:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = openai.chat.completions.create(
            model='llama_3_1_405b_instruct',
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1,
            stream=True
        )
        result = ""
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    result += delta
        return result.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()

    # Original tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS keywords (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        keyword TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS transcripts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        content TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS processed_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        timestamp TEXT,
                        transcript_id INTEGER,
                        notes_file TEXT,
                        quiz_file TEXT,
                        FOREIGN KEY (transcript_id) REFERENCES transcripts (id)
                    )''')

    # Check and add new columns if they don't exist
    try:
        cursor.execute("ALTER TABLE transcripts ADD COLUMN class_title TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE processed_files ADD COLUMN class_title TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # New tables for chat functionality
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE,
                        created_at TEXT,
                        last_active TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        message TEXT,
                        response TEXT,
                        timestamp TEXT,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                    )''')

    conn.commit()
    conn.close()


def save_keywords_to_db(keywords):
    """Save extracted keywords to database"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for keyword in keywords:
        cursor.execute("INSERT INTO keywords (timestamp, keyword) VALUES (?, ?)", (timestamp, keyword))
    conn.commit()
    conn.close()


def save_transcript_to_db(transcript, class_title=""):
    """Save transcript to database and return transcript ID"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO transcripts (timestamp, content, class_title) VALUES (?, ?, ?)",
                   (timestamp, transcript, class_title))
    transcript_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return transcript_id


def save_processed_file_info(filename, transcript_id, notes_file, quiz_file, class_title=""):
    """Save processed file information to database"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO processed_files (filename, timestamp, transcript_id, notes_file, quiz_file, class_title) VALUES (?, ?, ?, ?, ?, ?)",
        (filename, timestamp, transcript_id, notes_file, quiz_file, class_title))
    conn.commit()
    conn.close()


def get_all_transcripts():
    """Get all transcripts for chat context"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content, class_title, timestamp FROM transcripts ORDER BY timestamp")
    transcripts = cursor.fetchall()
    conn.close()
    return transcripts


def save_chat_message(session_id, message, response):
    """Save chat message to database"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Create session if doesn't exist
    cursor.execute("INSERT OR IGNORE INTO chat_sessions (session_id, created_at, last_active) VALUES (?, ?, ?)",
                   (session_id, timestamp, timestamp))

    # Update last active
    cursor.execute("UPDATE chat_sessions SET last_active = ? WHERE session_id = ?", (timestamp, session_id))

    # Save message
    cursor.execute("INSERT INTO chat_messages (session_id, message, response, timestamp) VALUES (?, ?, ?, ?)",
                   (session_id, message, response, timestamp))

    conn.commit()
    conn.close()


def get_chat_history(session_id):
    """Get chat history for a specific session"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT message, response, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
                   (session_id,))
    history = cursor.fetchall()
    conn.close()
    return history


def extract_keywords(text):
    """Extract keywords from text using TF-IDF"""
    if not text.strip():
        return ["No keywords found"]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    try:
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return keywords if len(keywords) > 0 else ["No significant keywords"]
    except ValueError:
        return ["No keywords found"]


def generate_quiz(transcript):
    """Generate quiz questions from transcript"""
    prompt = f"Create 5 MCQs and 3 short answer questions from this text with answers:\n\n{transcript}"
    return call_llm(prompt)


def generate_notes(transcript):
    """Generate comprehensive notes from transcript"""
    prompt = f"""
You are an academic assistant. Based on the following classroom lecture transcript, generate comprehensive student-friendly notes.
Include:
1. Topic name and class title
2. Key concepts and definitions
3. Important formulas or laws (if applicable)
4. Examples and explanations
5. Real-life applications
6. Summary points

Transcript:
{transcript}
"""
    return call_llm(prompt)


def save_pdf(content, filename):
    """Save content as PDF file with proper encoding handling"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Handle content encoding properly
    lines = content.split('\n')
    for line in lines:
        try:
            # Clean the line and handle encoding issues
            clean_line = line.strip()
            if not clean_line:
                pdf.ln(5)  # Add line break for empty lines
                continue

            # Try to encode to latin-1, replace problematic characters
            try:
                # First try direct encoding
                encoded_line = clean_line.encode('latin-1').decode('latin-1')
            except UnicodeEncodeError:
                # If that fails, replace problematic characters
                encoded_line = clean_line.encode('latin-1', errors='replace').decode('latin-1')

            # Add the line to PDF
            pdf.multi_cell(0, 10, encoded_line)

        except Exception as e:
            # If all else fails, skip the line but log it
            print(f"Skipping line due to encoding error: {str(e)}")
            continue

    try:
        pdf.output(filename)
        print(f"PDF saved successfully: {filename}")
    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        raise


# Alternative improved version using UTF-8 support
def save_pdf_utf8(content, filename):
    """Save content as PDF file with UTF-8 support"""
    from fpdf import FPDF

    class UTF8PDF(FPDF):
        def header(self):
            pass

        def footer(self):
            pass

    pdf = UTF8PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Try to add a UTF-8 compatible font
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.set_font('DejaVu', size=12)
    except:
        # Fallback to Arial
        pdf.set_font("Arial", size=12)

    lines = content.split('\n')
    for line in lines:
        try:
            clean_line = line.strip()
            if not clean_line:
                pdf.ln(5)
                continue

            # For UTF-8 support, try to handle the text properly
            if hasattr(pdf, 'add_font') and 'DejaVu' in str(pdf.fonts):
                # Use UTF-8 text directly
                pdf.multi_cell(0, 10, clean_line)
            else:
                # Fall back to latin-1 encoding
                encoded_line = clean_line.encode('latin-1', errors='replace').decode('latin-1')
                pdf.multi_cell(0, 10, encoded_line)

        except Exception as e:
            print(f"Error processing line: {str(e)}")
            continue

    try:
        pdf.output(filename)
        print(f"PDF saved successfully: {filename}")
    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        raise


# Additional helper function for directory and permissions
def ensure_output_directory(output_dir):
    """Ensure output directory exists with proper permissions"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Set permissions for Linux
        os.chmod(output_dir, 0o755)
        print(f"Output directory ready: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        raise


# Debug version to help identify the issue
def save_pdf_debug(content, filename):
    """Debug version of save_pdf to identify issues"""
    print(f"Creating PDF: {filename}")
    print(f"Content length: {len(content)}")
    print(f"Content preview: {content[:200]}...")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    lines = content.split('\n')
    print(f"Total lines to process: {len(lines)}")

    processed_lines = 0
    for i, line in enumerate(lines):
        try:
            clean_line = line.strip()
            if not clean_line:
                pdf.ln(5)
                continue

            print(f"Processing line {i + 1}: {clean_line[:50]}...")

            # Handle encoding
            encoded_line = clean_line.encode('latin-1', errors='replace').decode('latin-1')
            pdf.multi_cell(0, 10, encoded_line)
            processed_lines += 1

        except Exception as e:
            print(f"Error on line {i + 1}: {str(e)}")
            print(f"Problematic line: {repr(line)}")
            continue

    print(f"Successfully processed {processed_lines} lines")

    try:
        pdf.output(filename)
        print(f"PDF saved successfully: {filename}")

        # Verify file was created and has content
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"PDF file size: {file_size} bytes")
            if file_size < 1000:  # Very small PDF might indicate issues
                print("Warning: PDF file is very small, might be missing content")
        else:
            print("Error: PDF file was not created")

    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        raise


def convert_audio_to_wav(audio_file):
    """Convert audio file to WAV format for speech recognition"""
    try:
        audio = AudioSegment.from_file(audio_file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        raise Exception(f"Error converting audio: {str(e)}")


def transcribe_audio(audio_file):
    """Transcribe audio file to text"""
    recognizer = sr.Recognizer()
    try:
        wav_audio = convert_audio_to_wav(audio_file)
        with sr.AudioFile(wav_audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def get_all_classes():
    """Get list of all processed classes"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pf.class_title, pf.timestamp, pf.notes_file, pf.quiz_file, 
               t.content, pf.filename
        FROM processed_files pf
        JOIN transcripts t ON pf.transcript_id = t.id
        ORDER BY pf.timestamp DESC
    """)
    classes = cursor.fetchall()
    conn.close()

    result = []
    for class_data in classes:
        result.append({
            'class_title': class_data[0] or 'Untitled Class',
            'date': class_data[1],
            'notes_file': os.path.basename(class_data[2]),
            'quiz_file': os.path.basename(class_data[3]),
            'transcript_preview': class_data[4][:200] + "..." if len(class_data[4]) > 200 else class_data[4],
            'original_filename': class_data[5]
        })

    return result


def search_classes(query):
    """Search through class content"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pf.class_title, pf.timestamp, t.content, pf.notes_file, pf.quiz_file
        FROM processed_files pf
        JOIN transcripts t ON pf.transcript_id = t.id
        WHERE LOWER(t.content) LIKE ? OR LOWER(pf.class_title) LIKE ?
        ORDER BY pf.timestamp DESC
    """, (f'%{query}%', f'%{query}%'))

    results = cursor.fetchall()
    conn.close()

    search_results = []
    for result in results:
        # Find context around the search term
        content = result[2].lower()
        query_pos = content.find(query)
        if query_pos != -1:
            start = max(0, query_pos - 100)
            end = min(len(content), query_pos + 100)
            context = content[start:end]
        else:
            context = content[:200]

        search_results.append({
            'class_title': result[0] or 'Untitled Class',
            'date': result[1],
            'context': context + "...",
            'notes_file': os.path.basename(result[3]),
            'quiz_file': os.path.basename(result[4])
        })

    return search_results


def get_system_stats():
    """Get system statistics"""
    conn = sqlite3.connect("classroom_data.db")
    cursor = conn.cursor()

    # Get total classes
    cursor.execute("SELECT COUNT(*) FROM processed_files")
    total_classes = cursor.fetchone()[0]

    # Get total chat messages
    cursor.execute("SELECT COUNT(*) FROM chat_messages")
    total_messages = cursor.fetchone()[0]

    # Get recent activity
    cursor.execute("SELECT COUNT(*) FROM processed_files WHERE timestamp > date('now', '-7 days')")
    recent_classes = cursor.fetchone()[0]

    conn.close()

    return {
        'total_classes': total_classes,
        'total_chat_messages': total_messages,
        'recent_classes': recent_classes,
        'system_status': 'healthy'
    }


def process_audio_file(audio_file, class_title=""):
    """Main function to process audio file and generate content"""
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, audio_file.filename)
        audio_file.save(temp_file_path)

        # Transcribe audio
        transcript = transcribe_audio(temp_file_path)

        if len(transcript.strip().split()) < 5:
            raise Exception('Not enough content to analyze')

        # Extract keywords
        keywords = extract_keywords(transcript)

        # Save to database
        save_keywords_to_db(keywords)
        transcript_id = save_transcript_to_db(transcript, class_title)

        # Generate notes and quiz
        notes = generate_notes(transcript)
        quiz = generate_quiz(transcript)

        # Create output directory if it doesn't exist
        output_dir = "output_files"
        os.makedirs(output_dir, exist_ok=True)

        # Save as PDF files
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in class_title if c.isalnum() or c in (' ', '-', '_')).rstrip()

        notes_filename = f"{output_dir}/notes_{safe_title}_{timestamp}.pdf"
        quiz_filename = f"{output_dir}/quiz_{safe_title}_{timestamp}.pdf"

        save_pdf(f"Class Notes: {class_title}\n\n{notes}", notes_filename)
        save_pdf(f"Quiz: {class_title}\n\n{quiz}", quiz_filename)

        # Save processed file info
        save_processed_file_info(audio_file.filename, transcript_id, notes_filename, quiz_filename, class_title)

        # Clean up temp file
        os.remove(temp_file_path)
        os.rmdir(temp_dir)

        return {
            'success': True,
            'class_title': class_title,
            'transcript': transcript,
            'keywords': list(keywords),
            'notes': notes,
            'quiz': quiz,
            'files': {
                'notes_pdf': os.path.basename(notes_filename),
                'quiz_pdf': os.path.basename(quiz_filename)
            }
        }

    except Exception as e:
        # Clean up temp files in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise e


def generate_chat_response(message, session_id):
    """Generate chat response using LLM with class context"""
    # Get all transcripts for context
    transcripts = get_all_transcripts()

    # Build context from all class transcripts
    context = "You are a helpful academic assistant. You have access to the following class transcripts:\n\n"
    for i, (content, title, timestamp) in enumerate(transcripts):
        context += f"Class {i + 1}: {title} (Date: {timestamp})\n"
        context += f"Content: {content[:500]}...\n\n"

    context += """
Based on the above class content, please answer the student's question. If the question is about something not covered in class, mention that and still provide helpful information. Be educational and encouraging.
"""

    # Get chat history for better context
    chat_history = get_chat_history(session_id)
    if chat_history:
        context += "\n\nPrevious conversation:\n"
        for msg, resp, _ in chat_history[-3:]:  # Last 3 exchanges
            context += f"Student: {msg}\nAssistant: {resp}\n"

    # Generate response
    response = call_llm(f"Student question: {message}", context)

    # Save chat message
    save_chat_message(session_id, message, response)

    return response