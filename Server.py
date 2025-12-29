# ========================
# Standard Library Imports
# ========================
import base64
import os
import json
import time
import uuid
import tempfile
from datetime import datetime

# =========================
# Third-Party Library Imports
# =========================
import cv2
import torch
import whisper
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    UserMixin, login_user, logout_user, login_required, current_user, LoginManager
)

# ========================
# Local File Module Imports
# ========================
import ContentGeneration
from emotion_helper import (
    is_email, get_user_image_base64, allowed_file, allowed_image_file,
    process_video_emotions, save_face_image, create_user_hash, verify_password,
    format_analysis_data, format_user_data, format_user_profile,
    UPLOAD_FOLDER, REGISTERED_DIR, analyze_emotion_insights
)
from Voice_Chat_helper import (init_db, process_audio_file, generate_chat_response,
                               get_all_classes, search_classes, get_chat_history,
                               get_system_stats)

# Flask app initialization
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///made_with_hardwork.db'
app.config['SECRET_KEY'] = 'fdsfasdfsad34234sdfsd'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# File upload configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Create directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Language options
LANG_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Transcript file determination
translated_file = "translated_transcript.txt" if os.path.exists("translated_transcript.txt") else "translated.txt"

# Global storage for transcriptions
transcription_store = {}


# =============================================================================
# DATABASE MODELS
# =============================================================================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    face_image_path = db.Column(db.String(200))  # Path to stored face image
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class EmotionAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    top_emotion = db.Column(db.String(50))
    second_emotion = db.Column(db.String(50))
    distress_percentage = db.Column(db.Float)
    alert_triggered = db.Column(db.Boolean, default=False)
    chart_image = db.Column(db.Text)
    emotion_distribution = db.Column(db.Text)  # JSON string of emotion counts
    total_frames = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # New AI Insights fields
    recommendations = db.Column(db.Text)  # JSON string of recommendations
    positive_indicators = db.Column(db.Text)  # JSON string of positive indicators
    areas_to_watch = db.Column(db.Text)  # JSON string of areas to watch

    student = db.relationship('User', backref=db.backref('analyses', lazy=True))


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# =============================================================================
# AUTHENTICATION AND USER MANAGEMENT
# =============================================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/api/login', methods=['POST'])
def login():
    """Login route - accepts username or email with password"""
    try:
        # Get form data
        username_or_email = request.form.get('username_or_email') or request.form.get('username') or request.form.get(
            'email')
        password = request.form.get('password')

        # Validate input
        if not username_or_email or not password:
            return jsonify({'error': 'Username/email and password are required'}), 400

        # Determine if input is email or username
        if is_email(username_or_email):
            # Input is email
            user = User.query.filter_by(email=username_or_email).first()
        else:
            # Input is username
            user = User.query.filter_by(username=username_or_email).first()

        # Check if user exists and password is correct
        if not user or not verify_password(user.password, password):
            return jsonify({'error': 'Invalid username/email or password'}), 401

        # Log the user in
        login_user(user)

        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': format_user_profile(user)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500


@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    """Logout route"""
    try:
        logout_user()
        return jsonify({
            'status': 'success',
            'message': 'Logout successful'
        }), 200
    except Exception as e:
        return jsonify({'error': f'Logout failed: {str(e)}'}), 500


@app.route('/api/register', methods=['POST'])
def register():
    """User registration route"""
    try:
        # Handle form data (multipart/form-data)
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        face_image = request.files.get('face_image')

        # Validate input
        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password are required'}), 400

        # Validate email format
        if not is_email(email):
            return jsonify({'error': 'Invalid email format'}), 400

        if not face_image or not allowed_image_file(face_image.filename):
            return jsonify({'error': 'Valid face image (jpg, jpeg, png) is required'}), 400

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'error': 'Username already exists'}), 400

        # Check if the email already exists
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            return jsonify({'error': 'Email already exists'}), 400

        # Save face image
        face_image_path = save_face_image(face_image, username)

        # Create a new user
        hashed_password = create_user_hash(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            face_image_path=face_image_path
        )

        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'user': {
                'user_id': new_user.id,
                'username': new_user.username,
                'email': new_user.email
            }
        }), 201

    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


# =============================================================================
# CONTENT GENERATION ROUTES
# =============================================================================

@app.route('/api/transcribe', methods=['POST'])
def transcribe_video():
    print("Received request...")
    print("Files:", request.files)
    print("Form data:", request.form)

    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']

    # Check if filename is empty
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Reverse the LANG_OPTIONS dict: {'en': 'English', ...}
    LANG_CODES_TO_NAMES = {v: k for k, v in LANG_OPTIONS.items()}

    # Get code from form-data (default to 'en' if not provided)
    target_lang_code = request.form.get('target_language', 'en')

    # Convert code to language name
    if target_lang_code not in LANG_CODES_TO_NAMES:
        return jsonify({
            "error": f"Invalid target language code. Supported codes: {list(LANG_CODES_TO_NAMES.keys())}"
        }), 400

    target_lang = LANG_CODES_TO_NAMES[target_lang_code]

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        video_file.save(temp_file.name)
        video_path = temp_file.name

    try:
        # Extract audio
        audio_path = ContentGeneration.extract_audio(video_path)

        # Transcribe
        transcript = model.transcribe(audio_path)["text"]

        # Translate
        translated_text = ContentGeneration.translate_text(transcript, LANG_OPTIONS[target_lang])

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(audio_path)
        # Save original transcript
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        # Save translated transcript
        with open("translated_transcript.txt", "w", encoding="utf-8") as f:
            f.write(translated_text)

        # Return results as JSON
        return jsonify({
            "original_transcript": transcript,
            "translated_transcript": translated_text,
            "target_language": target_lang,

        })

    except Exception as e:
        # Clean up temporary files in case of error
        if os.path.exists(video_path):
            os.unlink(video_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)

        return jsonify({"error": str(e)}), 500


@app.route('/api/transcribelink', methods=['POST'])
def transcribe_link():
    youtube_link = request.form.get('youtube_link')
    target_language = request.form.get('target_language', 'en')

    if not youtube_link:
        return jsonify({"error": "YouTube link is required"}), 400

    if target_language not in ContentGeneration.LANG_MAP:
        return jsonify({"error": f"Unsupported language: {target_language}"}), 400

    try:
        # Step 1: Download audio
        audio_file = ContentGeneration.download_audio(youtube_link)
        if not audio_file:
            return jsonify({"error": "Failed to download audio from YouTube"}), 500

        # Step 2: Transcribe audio (auto-detect language)
        transcript_text, original_language = ContentGeneration.transcribe_audio_faster_whisper(audio_file)

        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

        if not transcript_text:
            return jsonify({"error": "Could not transcribe audio"}), 500

        # Step 3: Translate transcript (if target != original)
        if target_language != original_language:
            translated_transcript = ContentGeneration.translate_transcript(transcript_text, target_language)
        else:
            translated_transcript = transcript_text

        # Step 4: Generate notes
        detailed_notes = ContentGeneration.generate_detailed_notes(translated_transcript, target_language)

        # Step 5: Generate UUID and store
        transcription_id = str(uuid.uuid4())
        transcription_store[transcription_id] = {
            "original_transcript": transcript_text,
            "original_language": original_language,
            "translated_transcript": translated_transcript,
            "target_language": target_language,
            "detailed_notes": detailed_notes
        }

        # Step 6: Save files
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript_text)

        with open("translated_transcript.txt", "w", encoding="utf-8") as f:
            f.write(translated_transcript)

        with open("detailed_notes.txt", "w", encoding="utf-8") as f:
            f.write(detailed_notes)

        return jsonify({
            "transcription_id": transcription_id,
            "original_transcript": transcript_text,
            "original_language": original_language,
            "translated_transcript": translated_transcript,
            "target_language": target_language,
            "detailed_notes": detailed_notes,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/summary', methods=['GET'])
def generate_summary():
    """Generate and return summary."""
    try:
        # Get the latest transcript
        transcript_file = ContentGeneration.get_latest_transcript()

        # Load text from transcript
        with open(transcript_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Detect language
        detected_lang = ContentGeneration.detect_language(text)

        # Generate structured summary in bullet points
        summary_text = ContentGeneration.summarize_text(text, detected_lang)

        # Save Summary
        with open("summary.txt", "w", encoding="utf-8") as file:
            file.write(summary_text)

        return jsonify({
            "summary": summary_text,
            "language": detected_lang,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/flashcards', methods=['GET'])
def generate_flashcards_route():
    """Generate and return flashcards."""
    try:
        # Get the latest transcript
        transcript_file = ContentGeneration.get_latest_transcript()

        # Load text from transcript
        with open(transcript_file, "r", encoding="utf-8") as file:
            text = file.read()

        # Detect language
        detected_lang = ContentGeneration.detect_language(text)

        # Generate structured summary in bullet points
        summary_text = ContentGeneration.summarize_text(text, detected_lang)

        # Generate flashcards
        flashcards = ContentGeneration.generate_flashcards(summary_text)

        # Save Flashcards
        with open("flashcards.txt", "w", encoding="utf-8") as file:
            file.write(flashcards)
        flashcard_list = [line.strip() for line in flashcards.strip().split("\n") if line.strip()]

        return jsonify({
            "flashcards": flashcard_list,
            "language": detected_lang,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/quiz', methods=['GET'])
def quiz_route():
    transcript_text = ContentGeneration.read_transcript()

    if not transcript_text:
        return jsonify({"error": "No translated transcript file found"}), 404

    quiz = ContentGeneration.generate_quiz(transcript_text)
    return quiz, 200, {'Content-Type': 'application/json'}


@app.route('/api/exercise', methods=['GET'])
def exercise_route():
    transcript_text = ContentGeneration.read_transcript()

    if not transcript_text:
        return jsonify({"error": "No translated transcript file found"}), 404

    try:
        raw_text = ContentGeneration.generate_exercises(transcript_text)
        structured = ContentGeneration.parse_exercise_response(raw_text)

        if structured:
            return jsonify(structured)
        else:
            return jsonify({"error": "Failed to parse exercise response"}), 500
    except Exception as e:
        print("Exercise route error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route('/download/<transcription_id>/<file_type>', methods=['GET'])
def download_file_transcript(transcription_id, file_type):
    if transcription_id not in transcription_store:
        return jsonify({"error": "Transcription not found"}), 404

    transcription_data = transcription_store[transcription_id]

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
            if file_type == 'original':
                tmp.write(transcription_data["original_transcript"])
                filename = f"original_transcript_{transcription_data['original_language']}.txt"
            elif file_type == 'translated':
                tmp.write(transcription_data["translated_transcript"])
                filename = f"translated_transcript_{transcription_data['target_language']}.txt"
            elif file_type == 'notes':
                tmp.write(transcription_data["detailed_notes"])
                filename = f"detailed_notes_{transcription_data['target_language']}.txt"
            else:
                return jsonify({"error": "Invalid file type"}), 400

            tmp_path = tmp.name

        # Send the file
        return send_file(tmp_path, as_attachment=True, download_name=filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-note", methods=["POST"])
def generate_note():
    if 'file' not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files['file']
    target_language = request.form.get('target_language', 'en')
    filename = file.filename

    if not filename.lower().endswith(('.pdf', '.docx')):
        return jsonify({"error": "Unsupported file format"}), 400

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        # Extract text
        if filename.lower().endswith(".pdf"):
            notes_text = ContentGeneration.extract_text_from_pdf(temp_path)
        else:
            notes_text = ContentGeneration.extract_text_from_docx(temp_path)

        if not notes_text.strip():
            return jsonify({"error": "No text extracted from the file"}), 400

        # Translate notes
        translated_notes = ContentGeneration.translate_notes(notes_text, target_language)

        # Save to file
        ContentGeneration.save_text_to_file(translated_notes, "translated_notes.txt")

        return jsonify({
            "original_notes": notes_text,
            "translated_notes": translated_notes,
            "message": "Notes translated and saved to translated_notes.txt"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(temp_path)


# =============================================================================
# Emotion Analysis
# =============================================================================

@app.route("/api/emotion_dashboard", methods=["GET"])
def dashboard():
    """Dashboard route showing all analysis data with AI insights"""
    try:
        # Get all analyses with user data
        analyses = db.session.query(EmotionAnalysis, User).join(User).all()

        dashboard_data = []
        for analysis, user in analyses:
            dashboard_data.append(format_analysis_data(analysis, user))

        return jsonify({
            'status': 'success',
            'data': dashboard_data,
            'total_analyses': len(dashboard_data)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to fetch dashboard data: {str(e)}'}), 500


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload and process video for emotion analysis with AI insights"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Process video and analyze emotions with AI insights
            results = process_video_emotions(filepath, User, EmotionAnalysis, db)

            return jsonify({
                "message": "Video processed successfully. Analysis with AI insights saved to database.",
                "results": results,
                "total_people_analyzed": len(results)
            }), 200

        return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500


@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all registered users"""
    try:
        users = User.query.all()
        user_list = [format_user_data(user) for user in users]

        return jsonify({
            'status': 'success',
            'users': user_list,
            'total_users': len(user_list)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to fetch users: {str(e)}'}), 500


@app.route('/api/analysis/<int:user_id>', methods=['GET'])
def get_user_analysis(user_id):
    """Get analysis for a specific user with AI insights"""
    try:
        user = User.query.get_or_404(user_id)
        analysis = EmotionAnalysis.query.filter_by(student_id=user_id).first()

        if not analysis:
            return jsonify({'error': 'No analysis found for this user'}), 404

        # Parse emotion distribution
        emotion_dist = {}
        if analysis.emotion_distribution:
            try:
                emotion_dist = json.loads(analysis.emotion_distribution)
            except:
                emotion_dist = {}

        # Parse AI insights
        recommendations = []
        positive_indicators = []
        areas_to_watch = []

        if analysis.recommendations:
            try:
                recommendations = json.loads(analysis.recommendations)
            except:
                recommendations = []

        if analysis.positive_indicators:
            try:
                positive_indicators = json.loads(analysis.positive_indicators)
            except:
                positive_indicators = []

        if analysis.areas_to_watch:
            try:
                areas_to_watch = json.loads(analysis.areas_to_watch)
            except:
                areas_to_watch = []

        # Convert chart image to base64 string
        chart_image_b64 = None
        if analysis.chart_image:
            if isinstance(analysis.chart_image, bytes):
                chart_image_b64 = base64.b64encode(analysis.chart_image).decode('utf-8')
            else:
                chart_image_b64 = analysis.chart_image

        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            },
            'analysis': {
                'id': analysis.id,
                'top_emotion': analysis.top_emotion,
                'second_emotion': analysis.second_emotion,
                'distress_percentage': analysis.distress_percentage,
                'alert_triggered': analysis.alert_triggered,
                'emotion_distribution': emotion_dist,
                'total_frames': analysis.total_frames,
                'timestamp': analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'chart_image': chart_image_b64,
                'recommendations': recommendations,
                'positive_indicators': positive_indicators,
                'areas_to_watch': areas_to_watch
            }
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to fetch analysis: {str(e)}'}), 500


@app.route('/api/current_user', methods=['GET'])
@login_required
def get_current_user():
    """Get current logged in user info"""
    try:
        return jsonify({
            'status': 'success',
            'user': format_user_profile(current_user)
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to fetch current user: {str(e)}'}), 500


@app.route('/api/regenerate_insights/<int:user_id>', methods=['POST'])
def regenerate_insights(user_id):
    """Regenerate AI insights for a specific user"""
    try:
        user = User.query.get_or_404(user_id)
        analysis = EmotionAnalysis.query.filter_by(student_id=user_id).first()

        if not analysis:
            return jsonify({'error': 'No analysis found for this user'}), 404

        # Parse existing emotion distribution
        emotion_dist = {}
        if analysis.emotion_distribution:
            try:
                emotion_dist = json.loads(analysis.emotion_distribution)
            except:
                return jsonify({'error': 'Invalid emotion distribution data'}), 400

        # Generate new AI insights
        recommendations, positive_indicators, areas_to_watch = analyze_emotion_insights(
            emotion_dist, user.username, analysis.distress_percentage
        )

        if recommendations is None:
            return jsonify({'error': 'Failed to generate AI insights'}), 500

        # Update analysis with new insights
        analysis.recommendations = json.dumps(recommendations)
        analysis.positive_indicators = json.dumps(positive_indicators)
        analysis.areas_to_watch = json.dumps(areas_to_watch)

        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'AI insights regenerated successfully',
            'insights': {
                'recommendations': recommendations,
                'positive_indicators': positive_indicators,
                'areas_to_watch': areas_to_watch
            }
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to regenerate insights: {str(e)}'}), 500


@app.route('/api/batch_regenerate_insights', methods=['POST'])
def batch_regenerate_insights():
    """Regenerate AI insights for all users"""
    try:
        analyses = EmotionAnalysis.query.all()
        updated_count = 0

        for analysis in analyses:
            user = User.query.get(analysis.student_id)
            if not user:
                continue

            # Parse emotion distribution
            emotion_dist = {}
            if analysis.emotion_distribution:
                try:
                    emotion_dist = json.loads(analysis.emotion_distribution)
                except:
                    continue

            # Generate new AI insights
            recommendations, positive_indicators, areas_to_watch = analyze_emotion_insights(
                emotion_dist, user.username, analysis.distress_percentage
            )

            if recommendations is not None:
                analysis.recommendations = json.dumps(recommendations)
                analysis.positive_indicators = json.dumps(positive_indicators)
                analysis.areas_to_watch = json.dumps(areas_to_watch)
                updated_count += 1

        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': f'AI insights regenerated for {updated_count} users',
            'updated_count': updated_count
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to batch regenerate insights: {str(e)}'}), 500


# =============================================================================
# Voice Assistant And Chatbot
# =============================================================================
@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """Main endpoint for teachers to upload class recordings"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio_file']
        class_title = request.form.get('class_title', '')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process the audio file using voice_helper
        result = process_audio_file(file, class_title)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat_with_assistant():
    """Chat endpoint for students to ask questions about class content"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', f"session_{int(time.time())}")

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Generate response using voice_helper
        response = generate_chat_response(message, session_id)

        return jsonify({
            'success': True,
            'message': message,
            'response': response,
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of all processed classes for students to view"""
    try:
        classes = get_all_classes()
        return jsonify({'classes': classes})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/pdf/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated PDF files"""
    try:
        file_path = f"output_files/{filename}"
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-classes', methods=['POST'])
def search_classes_endpoint():
    """Search through class content"""
    try:
        data = request.get_json()
        query = data.get('query', '').lower()

        if not query:
            return jsonify({'error': 'Search query required'}), 400

        search_results = search_classes(query)
        return jsonify({'results': search_results, 'query': query})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-chat-history/<session_id>', methods=['GET'])
def get_chat_history_api(session_id):
    """Get chat history for a session"""
    try:
        history = get_chat_history(session_id)
        formatted_history = []
        for msg, resp, timestamp in history:
            formatted_history.append({
                'message': msg,
                'response': resp,
                'timestamp': timestamp
            })

        return jsonify({'history': formatted_history, 'session_id': session_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = get_system_stats()
        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        init_db()
        db.create_all()
    app.run(host='0.0.0.0', port=5009)
