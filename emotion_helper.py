import os
import cv2
import base64
import io
import json
import re
from collections import defaultdict, Counter
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from pytz import timezone
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Config
UPLOAD_FOLDER = "uploads"
REGISTERED_DIR = "registered_faces"
ALLOWED_EXTENSIONS = {'mp4'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
alert_emotions = ["sad", "angry", "fear"]
distress_threshold = 20  # in %

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REGISTERED_DIR, exist_ok=True)

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def is_email(text):
    """Check if the input text is an email address"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, text) is not None


def get_user_image_base64(user):
    """Get user's face image as base64 string"""
    if not user.face_image_path or not os.path.exists(user.face_image_path):
        return None

    try:
        with open(user.face_image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error reading user image: {e}")
        return None


def enhance_contrast(img):
    """Enhance contrast of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl1 = clahe.apply(gray)
    return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)


def allowed_file(filename):
    """Check if file extension is allowed for video files"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image_file(filename):
    """Check if file extension is allowed for image files"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def recognize_face(face_img):
    """Recognize face using DeepFace"""
    try:
        result = DeepFace.find(img_path=face_img, db_path=REGISTERED_DIR, enforce_detection=False, silent=True)
        if len(result) > 0 and not result[0].empty:
            identity = os.path.basename(result[0].iloc[0]['identity']).split(".")[0]
            return identity
    except:
        pass
    return None


def analyze_emotion(face_img):
    """Analyze emotion using DeepFace"""
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return None


def generate_pie_chart(emotion_counts, username):
    """Generate pie chart and return as base64 encoded string"""
    if not emotion_counts:
        return None

    labels = list(emotion_counts.keys())
    sizes = [emotion_counts[e] for e in labels]

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f"{username}'s Emotion Distribution", fontsize=14, fontweight='bold')

    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)

    # Convert to base64 STRING
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    img_buffer.close()

    return img_base64


def analyze_emotion_insights(emotion_counts, username, distress_percentage):
    """Generate AI-powered insights for emotion analysis"""
    if not emotion_counts:
        return None, None, None

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Calculate total emotions
        total_emotions = sum(emotion_counts.values())
        emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_counts.items()}

        # Create detailed prompt for analysis
        prompt = f"""
        Analyze the emotion data for student {username} and provide structured insights:

        Emotion Distribution:
        {json.dumps(emotion_percentages, indent=2)}

        Distress Level: {distress_percentage:.1f}%

        Based on this data, provide analysis in the following JSON format:
        {{
            "recommendations": [
                "specific actionable recommendation 1",
                "specific actionable recommendation 2",
                "specific actionable recommendation 3"
            ],
            "positive_indicators": [
                "positive aspect 1",
                "positive aspect 2"
            ],
            "areas_to_watch": [
                "area of concern 1",
                "area of concern 2"
            ]
        }}

        Focus on:
        - Educational context and classroom behavior
        - Actionable steps for teachers/counselors
        - Specific emotion patterns and their implications
        - Student wellbeing and academic performance
        """

        response = model.generate_content(prompt)

        # Extract JSON from response
        response_text = response.text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            insights = json.loads(json_str)

            return (
                insights.get('recommendations', []),
                insights.get('positive_indicators', []),
                insights.get('areas_to_watch', [])
            )
        else:
            print(f"Failed to extract JSON from AI response for {username}")
            return None, None, None

    except Exception as e:
        print(f"Error generating insights for {username}: {e}")
        return None, None, None


def get_top_emotions(emotion_counts):
    """Get top 2 emotions from emotion counts"""
    if not emotion_counts:
        return None, None

    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    top_emotion = sorted_emotions[0][0] if len(sorted_emotions) > 0 else None
    second_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None

    return top_emotion, second_emotion


def generate_debug_report(emotion_counts, total_frames):
    """Generate debug report in emotion.txt file (keeping for debug purposes)"""
    try:
        with open("emotion.txt", "w", encoding="utf-8") as f:
            f.write("=== EMOTION ANALYSIS DEBUG REPORT ===\n")
            f.write(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write("=" * 50 + "\n\n")

            if not emotion_counts:
                f.write("No emotion data found.\n")
                return

            for person, counts in emotion_counts.items():
                frames = total_frames[person]
                if frames == 0:
                    continue

                distress_frames = sum(counts[e] for e in alert_emotions)
                distress_percent = (distress_frames / frames) * 100

                f.write(f"👤 USER: {person}\n")
                f.write(f"• Total Frames Analyzed: {frames}\n")
                f.write(f"• Emotion Distribution: {dict(counts)}\n")
                f.write(f"• Distress Level: {distress_percent:.2f}%\n")
                f.write(f"• Alert Emotions Count: {distress_frames}\n")

                if distress_percent >= distress_threshold:
                    f.write(f"🚨 ALERT: {person} is showing signs of distress!\n")
                else:
                    f.write(f"✅ {person} appears to be in a stable emotional state.\n")

                # Most dominant emotion
                if counts:
                    most_common = counts.most_common(1)[0]
                    f.write(f"• Most Dominant Emotion: {most_common[0]} ({most_common[1]} occurrences)\n")

                f.write("-" * 40 + "\n\n")

            f.write("=== END OF REPORT ===\n")

        print("DEBUG: emotion.txt file generated successfully!")

    except Exception as e:
        print(f"DEBUG: Error generating emotion.txt: {str(e)}")


def process_video_emotions(filepath, User, EmotionAnalysis, db):
    """Process video file and analyze emotions with AI insights"""
    emotion_counts = defaultdict(Counter)
    total_frames = defaultdict(int)

    cap = cv2.VideoCapture(filepath)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 5 != 0:
            continue  # Skip every 4 out of 5 frames for speed

        enhanced = enhance_contrast(frame)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = enhanced[y:y + h, x:x + w]

            temp_path = os.path.join(UPLOAD_FOLDER, f"temp_face_{frame_idx}.jpg")
            cv2.imwrite(temp_path, face)

            name = recognize_face(temp_path)
            if name:
                try:
                    emotion = analyze_emotion(temp_path)
                    if emotion:
                        emotion_counts[name][emotion] += 1
                        total_frames[name] += 1
                except Exception as e:
                    print(f"Emotion error: {e}")

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    cap.release()
    os.remove(filepath)

    # Generate debug report
    generate_debug_report(emotion_counts, total_frames)

    # Save results to database
    results = []
    for person, counts in emotion_counts.items():
        # Find user by username
        user = User.query.filter_by(username=person).first()
        if not user:
            print(f"DEBUG: User '{person}' not found in database, skipping...")
            continue

        frames = total_frames[person]
        if frames == 0:
            print(f"DEBUG: No frames processed for user '{person}', skipping...")
            continue

        # Calculate distress percentage
        distress_frames = sum(counts[e] for e in alert_emotions)
        distress_percent = (distress_frames / frames) * 100

        # Get top emotions
        top_emotion, second_emotion = get_top_emotions(dict(counts))

        # Generate pie chart
        chart_image_b64 = generate_pie_chart(dict(counts), person)

        # Generate AI insights
        recommendations, positive_indicators, areas_to_watch = analyze_emotion_insights(
            dict(counts), person, distress_percent
        )

        print(f"DEBUG: Processing user '{person}' (ID: {user.id})")
        print(f"DEBUG: Emotion counts: {dict(counts)}")
        print(f"DEBUG: Total frames: {frames}")
        print(f"DEBUG: Distress percentage: {distress_percent:.2f}%")
        print(f"DEBUG: Top emotion: {top_emotion}, Second emotion: {second_emotion}")
        print(f"DEBUG: Alert triggered: {distress_percent >= distress_threshold}")

        # Check if analysis already exists for this user
        existing_analysis = EmotionAnalysis.query.filter_by(student_id=user.id).first()

        if existing_analysis:
            print(f"DEBUG: Updating existing analysis for user '{person}'")
            # Update existing analysis
            existing_analysis.top_emotion = top_emotion
            existing_analysis.second_emotion = second_emotion
            existing_analysis.distress_percentage = distress_percent
            existing_analysis.alert_triggered = distress_percent >= distress_threshold
            existing_analysis.chart_image = chart_image_b64
            existing_analysis.emotion_distribution = json.dumps(dict(counts))
            existing_analysis.total_frames = frames
            existing_analysis.timestamp = datetime.now(timezone('Asia/Kolkata'))
            # Update AI insights
            existing_analysis.recommendations = json.dumps(recommendations) if recommendations else None
            existing_analysis.positive_indicators = json.dumps(positive_indicators) if positive_indicators else None
            existing_analysis.areas_to_watch = json.dumps(areas_to_watch) if areas_to_watch else None
        else:
            print(f"DEBUG: Creating new analysis for user '{person}'")
            # Create new analysis
            new_analysis = EmotionAnalysis(
                student_id=user.id,
                top_emotion=top_emotion,
                second_emotion=second_emotion,
                distress_percentage=distress_percent,
                alert_triggered=distress_percent >= distress_threshold,
                chart_image=chart_image_b64,
                emotion_distribution=json.dumps(dict(counts)),
                total_frames=frames,
                recommendations=json.dumps(recommendations) if recommendations else None,
                positive_indicators=json.dumps(positive_indicators) if positive_indicators else None,
                areas_to_watch=json.dumps(areas_to_watch) if areas_to_watch else None
            )
            db.session.add(new_analysis)

        results.append({
            'username': person,
            'user_id': user.id,
            'emotion_counts': dict(counts),
            'total_frames': frames,
            'distress_percentage': distress_percent,
            'alert_triggered': distress_percent >= distress_threshold,
            'top_emotion': top_emotion,
            'second_emotion': second_emotion,
            'recommendations': recommendations,
            'positive_indicators': positive_indicators,
            'areas_to_watch': areas_to_watch
        })

    db.session.commit()
    print(f"DEBUG: Database updated successfully. Total people analyzed: {len(results)}")

    return results


def save_face_image(face_image, username):
    """Save uploaded face image to registered_faces directory"""
    face_image_filename = f"{username}.jpg"
    face_image_path = os.path.join(REGISTERED_DIR, face_image_filename)
    face_image.save(face_image_path)
    return face_image_path


def create_user_hash(password):
    """Create password hash"""
    return generate_password_hash(password)


def verify_password(password_hash, password):
    """Verify password against hash"""
    return check_password_hash(password_hash, password)


def format_analysis_data(analysis, user):
    """Format analysis data for API response with AI insights"""
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

    # Convert chart image to base64 string for display
    chart_image_b64 = None
    if analysis.chart_image:
        if isinstance(analysis.chart_image, bytes):
            # If it's bytes, encode to base64
            chart_image_b64 = base64.b64encode(analysis.chart_image).decode('utf-8')
        else:
            # If it's already a string, use as-is
            chart_image_b64 = analysis.chart_image

    return {
        'id': analysis.id,
        'username': user.username,
        'user_id': user.id,
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


def format_user_data(user):
    """Format user data for API response"""
    return {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'has_face_image': bool(user.face_image_path)
    }


def format_user_profile(user):
    """Format user profile with image for API response"""
    user_image = get_user_image_base64(user)

    return {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'user_image': user_image,
        'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S')
    }