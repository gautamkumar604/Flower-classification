"""
Flower Classification Model Training Script
This script trains a flower classification model using transfer learning with MobileNetV2.
Features:
- Fine-tuning with frozen base layers
- Data augmentation layers
- Proper preprocessing pipeline
- Model and class names persistence
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import json
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5
LEARNING_RATE = 0.0001
MODEL_SAVE_DIR = '../backend/model'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'flower_model.h5')
CLASS_NAMES_PATH = os.path.join(MODEL_SAVE_DIR, 'class_names.json')

# Flower classes
FLOWER_CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

print("=" * 60)
print("Flower Classification Model Training")
print("=" * 60)
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Classes: {FLOWER_CLASSES}")
print("=" * 60)


def create_data_augmentation():
    """Create data augmentation pipeline"""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")


def create_model():
    """
    Create a flower classification model using MobileNetV2 transfer learning with fine-tuning
    Preprocessing is applied via dataset.map() during training
    """
    # Load pre-trained MobileNetV2 model without top layers
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Enable fine-tuning: allow all layers to be trainable
    base_model.trainable = True
    
    # Freeze first 100 layers for stable training
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Create new model with data augmentation
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=IMG_SIZE + (3,)),
        
        # Data augmentation (applied on training data)
        create_data_augmentation(),
        
        # Pre-trained base model (preprocessed inputs come from dataset.map())
        base_model,
        
        # Classification layers
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model


def prepare_data(data_dir):
    """
    Prepare training and validation datasets using tf.data API with proper preprocessing
    """
    print("\n📁 Loading dataset from:", data_dir)
    
    # Load entire dataset with integer-encoded labels
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels='inferred',
        class_names=FLOWER_CLASSES,
        shuffle=True
    )
    
    # Get dataset size
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    
    # Split into training and validation
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Apply MobileNetV2 preprocessing to both datasets
    train_dataset = train_dataset.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize pipeline
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Count samples
    train_samples = train_size * BATCH_SIZE
    val_samples = (dataset_size - train_size) * BATCH_SIZE
    
    return train_dataset, val_dataset, train_samples, val_samples


def resolve_dataset_dir(data_dir):
    """Resolve the dataset directory, supporting both root and nested `flowers/` paths."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))
    class_dirs = [os.path.isdir(os.path.join(data_dir, cls)) for cls in FLOWER_CLASSES]
    if all(class_dirs):
        return data_dir

    alt_dir = os.path.join(data_dir, 'flowers')
    alt_class_dirs = [os.path.isdir(os.path.join(alt_dir, cls)) for cls in FLOWER_CLASSES]
    if all(alt_class_dirs):
        return alt_dir

    return None


def train_model(data_dir):
    """
    Train the flower classification model
    """
    print("\n🌸 Starting Flower Classification Model Training 🌸\n")
    
    # Resolve dataset directory
    resolved_dir = resolve_dataset_dir(data_dir)
    if resolved_dir is None:
        print(f"❌ Error: Dataset directory '{data_dir}' not found or missing class subfolders!")
        print("\nPlease organize your dataset in one of these structures:")
        print("dataset/")
        print("  ├── daisy/")
        print("  ├── dandelion/")
        print("  ├── rose/")
        print("  ├── sunflower/")
        print("  └── tulip/")
        print("or")
        print("dataset/")
        print("  └── flowers/")
        print("      ├── daisy/")
        print("      ├── dandelion/")
        print("      ├── rose/")
        print("      ├── sunflower/")
        print("      └── tulip/")
        return None, None
    
    # Prepare data
    print("\n📁 Preparing dataset...")
    train_dataset, val_dataset, train_samples, val_samples = prepare_data(resolved_dir)
    
    print(f"✓ Training samples: {train_samples}")
    print(f"✓ Validation samples: {val_samples}")
    
    # Create model
    print("\n🔨 Building model...")
    model = create_model()
    
    # Compile model with sparse_categorical_crossentropy for integer-encoded labels
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("\n🚀 Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"\n✓ Validation Loss: {val_loss:.4f}")
    print(f"✓ Validation Accuracy: {val_accuracy:.4f}")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Save model
    print(f"\n💾 Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("✓ Model saved successfully!")
    
    # Save class names
    print(f"💾 Saving class names to {CLASS_NAMES_PATH}...")
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(FLOWER_CLASSES, f, indent=2)
    print("✓ Class names saved successfully!")
    
    return model, history

def test_prediction(model):
    """
    Test the trained model with a sample prediction
    """
    print("\n🧪 Testing model prediction...")
    
    # Create a dummy image
    test_image = np.random.rand(1, 224, 224, 3).astype('float32')
    
    # Apply MobileNetV2 preprocessing
    test_image = preprocess_input(test_image)
    
    # Make prediction
    predictions = model.predict(test_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    print(f"✓ Test prediction successful!")
    print(f"  Predicted class: {FLOWER_CLASSES[predicted_class_idx]}")
    print(f"  Confidence: {confidence:.4f}")


if __name__ == '__main__':
    # Dataset directory path
    DATASET_DIR = '../dataset'
    
    # Train the model
    model, history = train_model(DATASET_DIR)
    
    if model is not None:
        # Test the model
        test_prediction(model)
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"\nModel saved at: {MODEL_SAVE_PATH}")
        print(f"Class names saved at: {CLASS_NAMES_PATH}")
        print("You can now use this model with the Flask API.\n")
