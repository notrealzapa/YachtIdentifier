import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load the MobileNetV2 model but exclude the top (final dense) layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Make sure the base model is not trainable
for layer in base_model.layers:
    layer.trainable = False

# Build the actual model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(149, activation='softmax')  # 149 classes for the yachts
])
