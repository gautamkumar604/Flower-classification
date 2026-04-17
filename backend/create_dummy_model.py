# """
# Create a dummy model for testing the Flower Classification API
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# import os

# # Create a simple dummy model for testing
# model = keras.Sequential([
#     keras.Input(shape=(224, 224, 3)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(5, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Save the model
# model_path = 'flower_model.h5'
# model.save(model_path)
# print(f'Dummy model saved to {model_path}')

# # Test loading the model
# try:
#     loaded_model = keras.models.load_model(model_path)
#     print('✓ Model can be loaded successfully')
# except Exception as e:
#     print(f'✗ Error loading model: {e}')