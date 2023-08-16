from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/zhumazhanbalapanov/Desktop/AugmentedYachtImages'  
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

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


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(149, activation='softmax') 
])

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

EPOCHS = 20


early_stopping = EarlyStopping(monitor='val_loss', patience=5)

checkpoint = ModelCheckpoint(
    "best_model.h5", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto'
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

classes = list(train_generator.class_indices.keys())
print(classes)

model.save("my_model.h5")
