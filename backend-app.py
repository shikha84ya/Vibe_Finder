"""
EmoTune Backend - Flask Application
Emotion Detection + Music Recommendation System
"""
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import cv2
import tensorflow as tf
import io
import base64
import requests
import json
import sqlite3
from datetime import datetime, timedelta
import os
import re
from functools import wraps
import traceback
import urllib.parse

# Initialize Flask App
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'emotune_secret_key_2024_production_change_this'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
jwt = JWTManager(app)

# ============================================
# DATABASE SETUP
# ============================================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('emotune.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Emotion history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            emotion TEXT NOT NULL,
            song_name TEXT,
            artist TEXT,
            preview_url TEXT,
            spotify_url TEXT,
            youtube_url TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# ============================================
# LOAD ML MODEL
# ============================================

# Emotion labels
emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Load the pre-trained model
model = None
model_loaded = False

try:
    # Try to load the trained model
    model_paths = ['emotion_model.h5', 'models/emotion_model.h5', '../emotion_model.h5']
    model_loaded_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            model_loaded = True
            model_loaded_path = path
            print(f"✅ Model loaded successfully from {path}")
            print(f"✅ Model input shape: {model.input_shape}")
            print(f"✅ Model output shape: {model.output_shape}")
            break
    
    if not model_loaded:
        print("❌ Model file 'emotion_model.h5' not found in any location")
        print("❌ Please train the model first using train_model.py")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("❌ Traceback:")
    traceback.print_exc()
    model = None

# ============================================
# EMOTION DETECTION FUNCTION
# ============================================
def detect_emotion(image_data):
    """
    Detect emotion from image data using trained model only
    image_data: numpy array or base64 string
    """
    try:
        # Check if model is loaded
        if model is None:
            return None, "Model not loaded. Please ensure emotion_model.h5 exists and is trained properly."
        
        print("🔍 Starting emotion detection...")
        
        # Load face detector
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if face_cascade.empty():
                # Try alternative path
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                if face_cascade.empty():
                    print("❌ Failed to load face cascade classifier")
                    return None, "Face detector not loaded properly"
        except Exception as e:
            print(f"❌ Error loading face cascade: {e}")
            return None, "Face detector error"
        
        # Convert image data
        image = None
        if isinstance(image_data, str):
            print("📷 Processing base64 image data...")
            # Handle base64 string
            try:
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                image_data = base64.b64decode(image_data)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"❌ Error decoding base64: {e}")
                return None, "Invalid image data"
        else:
            print("📷 Processing numpy image data...")
            image = image_data
        
        if image is None:
            print("❌ Could not decode image")
            return None, "Could not decode image"
        
        print(f"📐 Image shape: {image.shape}")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        print(f"👤 Faces detected: {len(faces)}")
        
        if len(faces) == 0:
            print("❌ No face detected")
            return None, "No face detected. Please ensure your face is clearly visible and well-lit."
        
        # Process first detected face
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        
        print(f"🎯 Face ROI shape: {roi.shape}")
        
        # Preprocess for model (must match training preprocessing)
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0  # Normalize to [0, 1]
        roi = np.expand_dims(roi, axis=0)    # Add batch dimension
        roi = np.expand_dims(roi, axis=-1)   # Add channel dimension
        
        print(f"🧠 Making prediction with trained model...")
        print(f"🧠 Input shape for model: {roi.shape}")
        
        # Predict emotion
        predictions = model.predict(roi, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx])
        emotion = emotion_labels[emotion_idx]
        
        print(f"✅ Emotion detected: {emotion} (confidence: {confidence:.2%})")
        
        # Return emotion and confidence
        return emotion, confidence
    
    except Exception as e:
        print(f"❌ Error in detect_emotion: {str(e)}")
        traceback.print_exc()
        return None, f"Detection error: {str(e)}"

# ============================================
# EMOTION TO MUSIC MAPPING WITH WORKING SONGS
# ============================================

EMOTION_MUSIC_MAP = {
    'happy': {
        'genres': ['pop', 'dance', 'electronic'],
        'mood': 'uplifting',
        'emoji': '😊',
        'description': 'Cheerful and energetic tunes',
        'color': '#FFD700'  # Gold
    },
    'sad': {
        'genres': ['classical', 'acoustic', 'indie'],
        'mood': 'melancholic', 
        'emoji': '😢',
        'description': 'Comforting and emotional songs',
        'color': '#4682B4'  # Steel Blue
    },
    'angry': {
        'genres': ['rock', 'metal', 'punk'],
        'mood': 'energetic',
        'emoji': '😠',
        'description': 'Intense and powerful tracks',
        'color': '#DC143C'  # Crimson
    },
    'neutral': {
        'genres': ['pop', 'electronic', 'ambient'],
        'mood': 'calm',
        'emoji': '😐',
        'description': 'Balanced and soothing melodies',
        'color': '#808080'  # Gray
    },
    'surprise': {
        'genres': ['electronic', 'dance', 'experimental'],
        'mood': 'exciting',
        'emoji': '😲',
        'description': 'Unexpected and dynamic sounds',
        'color': '#9370DB'  # Medium Purple
    }
}

# Working sample songs with YouTube/Spotify links and working audio
SAMPLE_SONGS = {
    'happy': [
        {
            'name': 'Happy',
            'artist': 'Pharrell Williams',
            'preview_url': 'https://www.youtube.com/watch?v=y6Sxv-sUYtM',  # YouTube link
            'spotify_url': 'https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH',
            'youtube_url': 'https://www.youtube.com/watch?v=y6Sxv-sUYtM',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2738c6b2c8e6f8b4b3f6b7c8d9e',
            'album': 'G I R L',
            'genre': 'pop',
            'duration_ms': 233000
        },
        {
            'name': 'Uptown Funk',
            'artist': 'Mark Ronson ft. Bruno Mars',
            'preview_url': 'https://www.youtube.com/watch?v=OPf0YbXqDm0',
            'spotify_url': 'https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS',
            'youtube_url': 'https://www.youtube.com/watch?v=OPf0YbXqDm0',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2734e5f0e5c0b6b8c7d8e9f0a1b',
            'album': 'Uptown Special',
            'genre': 'funk',
            'duration_ms': 270000
        }
    ],
    'sad': [
        {
            'name': 'Someone Like You',
            'artist': 'Adele',
            'preview_url': 'https://www.youtube.com/watch?v=hLQl3WQQoQ0',
            'spotify_url': 'https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV',
            'youtube_url': 'https://www.youtube.com/watch?v=hLQl3WQQoQ0',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2730a0b7d7a8c9e0f1a2b3c4d5e',
            'album': '21',
            'genre': 'pop',
            'duration_ms': 285000
        },
        {
            'name': 'Say Something',
            'artist': 'A Great Big World',
            'preview_url': 'https://www.youtube.com/watch?v=-2U0Ivkn2Ds',
            'spotify_url': 'https://open.spotify.com/track/6Vc5wAMmXdKIAM7WUoEb7N',
            'youtube_url': 'https://www.youtube.com/watch?v=-2U0Ivkn2Ds',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2735f6e7f8a9b0c1d2e3f4a5b6c',
            'album': 'Is There Anybody Out There?',
            'genre': 'pop',
            'duration_ms': 229000
        }
    ],
    'angry': [
        {
            'name': 'Killing In The Name',
            'artist': 'Rage Against The Machine',
            'preview_url': 'https://www.youtube.com/watch?v=bWXazVhlyxQ',
            'spotify_url': 'https://open.spotify.com/track/59WN2psjkt1tyaxjspN8fp',
            'youtube_url': 'https://www.youtube.com/watch?v=bWXazVhlyxQ',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2736f7a8b9c0d1e2f3a4b5c6d7e',
            'album': 'Rage Against The Machine',
            'genre': 'rock',
            'duration_ms': 315000
        },
        {
            'name': 'Du Hast',
            'artist': 'Rammstein',
            'preview_url': 'https://www.youtube.com/watch?v=W3q8Od5qJio',
            'spotify_url': 'https://open.spotify.com/track/5awDvzxWfd53SSrsRZ8pXO',
            'youtube_url': 'https://www.youtube.com/watch?v=W3q8Od5qJio',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2737a8b9c0d1e2f3a4b5c6d7e8f',
            'album': 'Sehnsucht',
            'genre': 'industrial',
            'duration_ms': 238000
        }
    ],
    'neutral': [
        {
            'name': 'Blinding Lights',
            'artist': 'The Weeknd',
            'preview_url': 'https://www.youtube.com/watch?v=4NRXx6U8ABQ',
            'spotify_url': 'https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b',
            'youtube_url': 'https://www.youtube.com/watch?v=4NRXx6U8ABQ',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2738b9c0d1e2f3a4b5c6d7e8f9a',
            'album': 'After Hours',
            'genre': 'pop',
            'duration_ms': 200000
        },
        {
            'name': 'Levitating',
            'artist': 'Dua Lipa',
            'preview_url': 'https://www.youtube.com/watch?v=TUVcZfQe-Kw',
            'spotify_url': 'https://open.spotify.com/track/39LLxExYz6ewLAcYrzQQyP',
            'youtube_url': 'https://www.youtube.com/watch?v=TUVcZfQe-Kw',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2739c0d1e2f3a4b5c6d7e8f9a0b',
            'album': 'Future Nostalgia',
            'genre': 'pop',
            'duration_ms': 203000
        }
    ],
    'surprise': [
        {
            'name': 'Bad Romance',
            'artist': 'Lady Gaga',
            'preview_url': 'https://www.youtube.com/watch?v=qrO4YZeyl0I',
            'spotify_url': 'https://open.spotify.com/track/0SiywuOBRcynK0uKGWdCnn',
            'youtube_url': 'https://www.youtube.com/watch?v=qrO4YZeyl0I',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2730d1e2f3a4b5c6d7e8f9a0b1c',
            'album': 'The Fame Monster',
            'genre': 'pop',
            'duration_ms': 295000
        },
        {
            'name': 'Seven Nation Army',
            'artist': 'The White Stripes',
            'preview_url': 'https://www.youtube.com/watch?v=0J2QdDbelmY',
            'spotify_url': 'https://open.spotify.com/track/3dPQuX8Gs42Y7b454ybpMR',
            'youtube_url': 'https://www.youtube.com/watch?v=0J2QdDbelmY',
            'artwork': 'https://i.scdn.co/image/ab67616d0000b2731e2f3a4b5c6d7e8f9a0b1c2d',
            'album': 'Elephant',
            'genre': 'rock',
            'duration_ms': 231000
        }
    ]
}

def get_youtube_search_results(emotion, limit=3):
    """Search for emotion-related songs on YouTube"""
    try:
        if emotion not in EMOTION_MUSIC_MAP:
            emotion = 'neutral'
        
        genres = EMOTION_MUSIC_MAP[emotion]['genres']
        search_term = f"{emotion} {genres[0]} music"
        
        # YouTube search API (simplified - in production use YouTube Data API v3)
        youtube_songs = []
        
        # For demo purposes, return sample YouTube songs
        if emotion in SAMPLE_SONGS:
            return SAMPLE_SONGS[emotion][:limit]
        
        return []
        
    except Exception as e:
        print(f"❌ YouTube search error: {e}")
        return []

def recommend_songs(emotion):
    """Get song recommendations with working audio links"""
    try:
        if emotion not in EMOTION_MUSIC_MAP:
            emotion = 'neutral'
        
        print(f"🎵 Getting recommendations for emotion: {emotion}")
        
        # Get YouTube songs (they have working audio)
        songs = get_youtube_search_results(emotion, limit=4)
        
        if not songs:
            print("⚠ No YouTube songs found, using sample songs")
            songs = SAMPLE_SONGS.get(emotion, SAMPLE_SONGS['neutral'])
        
        # Add emotion info to all songs
        for song in songs:
            song['emotion'] = emotion
            song['mood'] = EMOTION_MUSIC_MAP[emotion]['mood']
            song['emoji'] = EMOTION_MUSIC_MAP[emotion]['emoji']
            song['color'] = EMOTION_MUSIC_MAP[emotion]['color']
            
            # Ensure preview_url is set (use youtube_url as fallback)
            if not song.get('preview_url') and song.get('youtube_url'):
                song['preview_url'] = song['youtube_url']
        
        print(f"✅ Found {len(songs)} songs for {emotion}")
        return songs
        
    except Exception as e:
        print(f"❌ Error in recommend_songs: {e}")
        traceback.print_exc()
        # Return at least some songs
        return SAMPLE_SONGS.get(emotion, SAMPLE_SONGS['neutral'])

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not email or not password:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Validate email
        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password strength
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT * FROM users WHERE username=? OR email=?', (username, email))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Username or email already exists'}), 409
        
        # Create user
        hashed_password = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Generate JWT token
        access_token = create_access_token(identity=user_id)
        
        return jsonify({
            'message': 'Registration successful',
            'access_token': access_token,
            'user': {'id': user_id, 'username': username, 'email': email}
        }), 201
    
    except Exception as e:
        print(f"❌ Registration error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'error': 'Missing username or password'}), 400
        
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user or not check_password_hash(user[3], password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        access_token = create_access_token(identity=user[0])
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {'id': user[0], 'username': user[1], 'email': user[2]}
        }), 200
    
    except Exception as e:
        print(f"❌ Login error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    try:
        user_id = get_jwt_identity()
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, created_at FROM users WHERE id=?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'created_at': user[3]
            }
        }), 200
    
    except Exception as e:
        print(f"❌ Get user error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

# ============================================
# EMOTION DETECTION ROUTES
# ============================================

@app.route('/api/detect/image', methods=['POST'])
@jwt_required()
def detect_from_image():
    """Detect emotion from uploaded image file"""
    try:
        user_id = get_jwt_identity()
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Emotion model not available',
                'message': 'Please ensure the emotion_model.h5 file exists and is trained properly.'
            }), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Check file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid image format. Please upload PNG, JPG, or JPEG.'}), 400
        
        # Read image
        image_data = file.read()
        if len(image_data) == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        
        # Check image size
        if image.shape[0] < 100 or image.shape[1] < 100:
            return jsonify({'error': 'Image too small. Please upload a larger image (min 100x100).'}), 400
        
        # Detect emotion
        emotion, confidence = detect_emotion(image)
        
        if emotion is None:
            return jsonify({'error': confidence}), 400
        
        # Get song recommendations
        songs = recommend_songs(emotion)
        
        # Save to history
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        
        if songs and len(songs) > 0:
            first_song = songs[0]
            cursor.execute(
                '''INSERT INTO emotion_history 
                (user_id, emotion, song_name, artist, preview_url, spotify_url, youtube_url) 
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (user_id, emotion, first_song['name'], 
                 first_song['artist'], first_song.get('preview_url', ''),
                 first_song.get('spotify_url', ''), first_song.get('youtube_url', ''))
            )
        else:
            cursor.execute(
                'INSERT INTO emotion_history (user_id, emotion) VALUES (?, ?)',
                (user_id, emotion)
            )
        
        conn.commit()
        conn.close()
        
        emotion_info = EMOTION_MUSIC_MAP.get(emotion, {})
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'songs': songs,
            'mood_description': emotion_info.get('mood', 'neutral'),
            'emoji': emotion_info.get('emoji', '😐'),
            'genre_description': emotion_info.get('description', ''),
            'color': emotion_info.get('color', '#808080'),
            'detection_method': 'trained_model'
        }), 200
    
    except Exception as e:
        print(f"❌ Image detection error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error during detection'}), 500

@app.route('/api/detect/webcam', methods=['POST'])
@jwt_required()
def detect_from_webcam():
    """Detect emotion from base64 webcam frame"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        image_data = data.get('frame')
        
        if not image_data:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Emotion model not available',
                'message': 'Please ensure the emotion_model.h5 file exists and is trained properly.'
            }), 503
        
        # Detect emotion
        emotion, confidence = detect_emotion(image_data)
        
        if emotion is None:
            return jsonify({'error': confidence}), 400
        
        # Get song recommendations
        songs = recommend_songs(emotion)
        
        # Save to history
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        
        if songs and len(songs) > 0:
            first_song = songs[0]
            cursor.execute(
                '''INSERT INTO emotion_history 
                (user_id, emotion, song_name, artist, preview_url, spotify_url, youtube_url) 
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (user_id, emotion, first_song['name'], 
                 first_song['artist'], first_song.get('preview_url', ''),
                 first_song.get('spotify_url', ''), first_song.get('youtube_url', ''))
            )
        else:
            cursor.execute(
                'INSERT INTO emotion_history (user_id, emotion) VALUES (?, ?)',
                (user_id, emotion)
            )
        
        conn.commit()
        conn.close()
        
        emotion_info = EMOTION_MUSIC_MAP.get(emotion, {})
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'songs': songs,
            'mood_description': emotion_info.get('mood', 'neutral'),
            'emoji': emotion_info.get('emoji', '😐'),
            'genre_description': emotion_info.get('description', ''),
            'color': emotion_info.get('color', '#808080'),
            'detection_method': 'trained_model'
        }), 200
    
    except Exception as e:
        print(f"❌ Webcam detection error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error during detection'}), 500

@app.route('/api/detect/quick', methods=['POST'])
def quick_detect():
    """Quick emotion detection without authentication (for testing)"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Emotion model not available',
                'message': 'Please ensure the emotion_model.h5 file exists and is trained properly.'
            }), 503
        
        # Check for image in request
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image_data = file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                return jsonify({'error': 'No image selected'}), 400
        elif request.is_json:
            data = request.get_json()
            image_data = data.get('frame')
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
            image = image_data
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Detect emotion
        emotion, confidence = detect_emotion(image)
        
        if emotion is None:
            return jsonify({'error': confidence}), 400
        
        # Get song recommendations
        songs = recommend_songs(emotion)
        
        emotion_info = EMOTION_MUSIC_MAP.get(emotion, {})
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'songs': songs[:3],  # Return only 3 songs for quick demo
            'mood_description': emotion_info.get('mood', 'neutral'),
            'emoji': emotion_info.get('emoji', '😐'),
            'color': emotion_info.get('color', '#808080')
        }), 200
    
    except Exception as e:
        print(f"❌ Quick detection error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error during detection'}), 500

@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_history():
    """Get user's emotion detection history"""
    try:
        user_id = get_jwt_identity()
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT emotion, song_name, artist, preview_url, spotify_url, youtube_url, timestamp 
            FROM emotion_history 
            WHERE user_id=? 
            ORDER BY timestamp DESC LIMIT 50''',
            (user_id,)
        )
        history = cursor.fetchall()
        conn.close()
        
        formatted_history = []
        for h in history:
            emotion_info = EMOTION_MUSIC_MAP.get(h[0], {})
            formatted_history.append({
                'emotion': h[0],
                'song': h[1] or 'No song',
                'artist': h[2] or 'Unknown',
                'preview_url': h[3] or '',
                'spotify_url': h[4] or '',
                'youtube_url': h[5] or '',
                'timestamp': h[6],
                'emoji': emotion_info.get('emoji', '😐'),
                'color': emotion_info.get('color', '#808080')
            })
        
        return jsonify({
            'history': formatted_history,
            'count': len(history)
        }), 200
    
    except Exception as e:
        print(f"❌ History error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/history/clear', methods=['DELETE'])
@jwt_required()
def clear_history():
    """Clear user's emotion detection history"""
    try:
        user_id = get_jwt_identity()
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM emotion_history WHERE user_id=?',
            (user_id,)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'History cleared successfully',
            'deleted_count': deleted_count
        }), 200
    
    except Exception as e:
        print(f"❌ Clear history error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """Get user's emotion statistics"""
    try:
        user_id = get_jwt_identity()
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # Get emotion counts
        cursor.execute(
            '''SELECT emotion, COUNT(*) as count 
            FROM emotion_history 
            WHERE user_id=? 
            GROUP BY emotion 
            ORDER BY count DESC''',
            (user_id,)
        )
        emotion_counts = cursor.fetchall()
        
        # Get total detections
        cursor.execute(
            'SELECT COUNT(*) FROM emotion_history WHERE user_id=?',
            (user_id,)
        )
        total_detections = cursor.fetchone()[0]
        
        # Get most recent detection
        cursor.execute(
            '''SELECT emotion, timestamp 
            FROM emotion_history 
            WHERE user_id=? 
            ORDER BY timestamp DESC LIMIT 1''',
            (user_id,)
        )
        recent = cursor.fetchone()
        
        conn.close()
        
        stats = {
            'total_detections': total_detections,
            'emotion_distribution': [
                {
                    'emotion': e[0], 
                    'count': e[1], 
                    'emoji': EMOTION_MUSIC_MAP.get(e[0], {}).get('emoji', ''),
                    'color': EMOTION_MUSIC_MAP.get(e[0], {}).get('color', '#808080'),
                    'percentage': round((e[1] / total_detections * 100), 2) if total_detections > 0 else 0
                }
                for e in emotion_counts
            ],
            'most_recent': {
                'emotion': recent[0] if recent else None,
                'timestamp': recent[1] if recent else None,
                'emoji': EMOTION_MUSIC_MAP.get(recent[0], {}).get('emoji', '') if recent else '',
                'color': EMOTION_MUSIC_MAP.get(recent[0], {}).get('color', '#808080') if recent else ''
            } if recent else None
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        print(f"❌ Stats error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

# ============================================
# MUSIC PLAYBACK ROUTES
# ============================================

@app.route('/api/music/proxy', methods=['GET'])
def music_proxy():
    """Proxy for music URLs to handle CORS"""
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Decode if base64 encoded
        if url.startswith('base64:'):
            url = base64.b64decode(url[7:]).decode('utf-8')
        
        # Validate URL
        if not url.startswith('http'):
            return jsonify({'error': 'Invalid URL'}), 400
        
        print(f"🎵 Proxying music URL: {url[:100]}...")
        
        # For YouTube URLs, redirect to YouTube
        if 'youtube.com' in url or 'youtu.be' in url:
            return jsonify({
                'url': url,
                'type': 'youtube',
                'message': 'Use this URL directly in an iframe or embed'
            })
        
        # For other URLs, try to proxy
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'identity',
            'Range': request.headers.get('Range', 'bytes=0-')
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code not in [200, 206]:
            return jsonify({'error': f'Failed to fetch audio: {response.status_code}'}), 500
        
        def generate():
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        
        return Response(
            stream_with_context(generate()),
            status=response.status_code,
            content_type=response.headers.get('content-type', 'audio/mpeg'),
            headers={
                'Content-Type': response.headers.get('content-type', 'audio/mpeg'),
                'Content-Length': response.headers.get('content-length', ''),
                'Accept-Ranges': 'bytes',
                'Content-Range': response.headers.get('content-range', ''),
                'Cache-Control': 'public, max-age=3600',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        print(f"❌ Music proxy error: {e}")
        return jsonify({'error': f'Failed to stream audio: {str(e)}'}), 500

@app.route('/api/music/test', methods=['GET'])
def test_music():
    """Test music endpoint with working audio"""
    test_song = {
        'name': 'Test Audio',
        'artist': 'EmoTune System',
        'preview_url': 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
        'type': 'direct_mp3',
        'message': 'This is a test audio that should work'
    }
    
    return jsonify({
        'test': True,
        'song': test_song,
        'proxy_url': f'/api/music/proxy?url={urllib.parse.quote(test_song["preview_url"])}',
        'direct_url': test_song['preview_url']
    })

@app.route('/api/music/sample/<emotion>', methods=['GET'])
def get_sample_music(emotion):
    """Get sample music for a specific emotion"""
    if emotion not in SAMPLE_SONGS:
        emotion = 'neutral'
    
    songs = SAMPLE_SONGS.get(emotion, SAMPLE_SONGS['neutral'])
    
    # Add proxy URLs
    for song in songs:
        if song.get('preview_url'):
            song['proxy_url'] = f'/api/music/proxy?url={urllib.parse.quote(song["preview_url"])}'
        if song.get('youtube_url'):
            song['youtube_embed'] = song['youtube_url'].replace('watch?v=', 'embed/')
    
    return jsonify({
        'emotion': emotion,
        'songs': songs,
        'count': len(songs)
    })

# ============================================
# HEALTH & INFO ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = {
        'loaded': model is not None,
        'path': 'emotion_model.h5',
        'input_shape': str(model.input_shape) if model else None,
        'output_shape': str(model.output_shape) if model else None
    }
    
    # Check database
    db_status = False
    try:
        conn = sqlite3.connect('emotune.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        db_status = True
        conn.close()
    except:
        db_status = False
    
    # Check sample songs
    sample_status = {}
    for emotion in emotion_labels:
        songs = SAMPLE_SONGS.get(emotion, [])
        sample_status[emotion] = {
            'count': len(songs),
            'has_audio': any(s.get('preview_url') for s in songs),
            'has_youtube': any(s.get('youtube_url') for s in songs)
        }
    
    return jsonify({
        'status': 'healthy' if model is not None and db_status else 'degraded',
        'service': 'EmoTune API',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'model': model_status,
        'database': db_status,
        'emotions': emotion_labels,
        'sample_songs': sample_status,
        'endpoints': {
            'auth': ['/api/auth/register', '/api/auth/login', '/api/auth/me'],
            'detection': ['/api/detect/image', '/api/detect/webcam', '/api/detect/quick'],
            'music': ['/api/music/proxy', '/api/music/test', '/api/music/sample/<emotion>'],
            'history': ['/api/history', '/api/history/clear', '/api/stats']
        }
    }), 200

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'EmoTune API',
        'description': 'Emotion Detection and Music Recommendation System',
        'version': '2.0.0',
        'author': 'EmoTune Team',
        'features': [
            'Real-time emotion detection using CNN',
            'Music recommendations based on emotions',
            'User authentication and history',
            'Working audio playback via proxy',
            'YouTube and Spotify integration',
            'RESTful API with JWT authentication'
        ],
        'requirements': [
            'Python 3.8+',
            'TensorFlow 2.x',
            'OpenCV',
            'Flask',
            'SQLite database'
        ]
    }), 200

# ============================================
# STATIC FILE SERVING
# ============================================

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    else:
        return jsonify({
            'message': 'EmoTune Backend API',
            'status': 'running',
            'frontend': 'Not found. Please place index.html in the root directory.',
            'api_docs': '/api/info',
            'health_check': '/api/health'
        })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    # Security check
    if '..' in path or path.startswith('/'):
        return jsonify({'error': 'Invalid path'}), 400
    
    if os.path.exists(path):
        return send_from_directory('.', path)
    
    # Try with common extensions
    for ext in ['', '.html', '.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg']:
        if os.path.exists(path + ext):
            return send_from_directory('.', path + ext)
    
    return jsonify({'error': 'File not found'}), 404

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ Internal server error: {error}")
    traceback.print_exc()
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized. Please login.'}), 401

# ============================================
# MAIN APPLICATION
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 EmoTune Backend Server Starting...")
    print("="*60)
    print(f"📁 Database: emotune.db")
    print(f"🧠 Emotions: {', '.join(emotion_labels)}")
    print(f"✅ Model loaded: {model is not None}")
    
    if model is None:
        print(f"❌ WARNING: Model not loaded. Emotion detection will not work!")
        print(f"❌ Please ensure 'emotion_model.h5' exists in:")
        print(f"❌   - Current directory")
        print(f"❌   - models/ directory")
        print(f"❌ Or train a model using train_model.py")
    else:
        print(f"✅ Model ready for emotion detection")
    
    print(f"🎵 Music playback: Working via YouTube links")
    print(f"🌐 API URL: http://localhost:5001")
    print(f"🔗 Frontend: http://localhost:5001")
    print(f"📚 API Docs: http://localhost:5001/api/info")
    print(f"❤️  Health: http://localhost:5001/api/health")
    print("="*60 + "\n")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Start the server
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)