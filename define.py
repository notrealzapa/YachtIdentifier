from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
base_dir = '/Users/zhumazhanbalapanov/Desktop/AugmentedYachtImages'  # Make sure to set your path here
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Create ImageDataGenerator with the desired augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of the data will be used for validation
)

# Create generators
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# Load MobileNetV2 model without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(149, activation='softmax')  # 149 neurons since you have 149 yacht classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Check the model's architecture
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Specify the number of epochs you want to train for
EPOCHS = 20

# Using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Save only the best model during training (with lowest validation loss)
checkpoint = ModelCheckpoint(
    "best_model.h5", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto'
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

classes = list(train_generator.class_indices.keys())
print(classes)

model.save("my_model.h5")
