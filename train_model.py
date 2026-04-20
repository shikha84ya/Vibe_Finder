"""
EmoTune - Model Training Script
Train CNN model on facial emotion dataset
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

DATASET_PATH = 'dataset'  # Your emotion dataset folder structure
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ============================================
# DATA LOADING
# ============================================

def load_data():
    """
    Load images from dataset folder
    Expected structure: dataset/emotion_name/image1.jpg, image2.jpg, ...
    """
    X = []
    y = []
    
    for idx, emotion in enumerate(EMOTIONS):
        emotion_path = os.path.join(DATASET_PATH, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"⚠ Warning: {emotion_path} not found")
            continue
        
        for filename in os.listdir(emotion_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Read image in grayscale
                    img_path = os.path.join(emotion_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Resize
                    img = cv2.resize(img, IMAGE_SIZE)
                    
                    # Normalize
                    img = img.astype('float32') / 255.0
                    
                    X.append(img)
                    y.append(idx)
                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return np.array(X), np.array(y)

# ============================================
# MODEL ARCHITECTURE
# ============================================

def create_model():
    """Create CNN model for emotion classification"""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                     input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(len(EMOTIONS), activation='softmax')
    ])
    
    return model

# ============================================
# TRAINING
# ============================================

def train_model():
    """Train the emotion recognition model"""
    print("Loading dataset...")
    X, y = load_data()
    
    if len(X) == 0:
        print("❌ No images found in dataset folder")
        print(f"Please organize your images as: {DATASET_PATH}/emotion_name/image.jpg")
        return
    
    print(f"✓ Loaded {len(X)} images")
    
    # Add channel dimension
    X = np.expand_dims(X, axis=-1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Create model
    model = create_model()
    model.summary()
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n🚀 Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Save model
    model.save('emotion_model.h5')
    print("✓ Model saved as 'emotion_model.h5'")
    
    return model, history

if __name__ == "__main__":
    train_model()