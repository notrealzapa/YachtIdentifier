import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/zhumazhanbalapanov/Desktop/AugmentedYachtImages'  

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training' 
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  
)

print(f"Total training samples: {train_generator.n}")
print(f"Total validation samples: {val_generator.n}")
