import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

data_dir = "/Users/zhumazhanbalapanov/Desktop/YachtImages"
save_dir = "/Users/zhumazhanbalapanov/Desktop/AugmentedYachtImages"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for yacht_name in os.listdir(data_dir):
    yacht_path = os.path.join(data_dir, yacht_name)
    
    if os.path.isdir(yacht_path):
        print(f"Directory: {yacht_path}")
        
        augmented_yacht_dir = os.path.join(save_dir, yacht_name)
        if not os.path.exists(augmented_yacht_dir):
            os.mkdir(augmented_yacht_dir)

        for image_file in os.listdir(yacht_path):
            if os.path.isfile(os.path.join(yacht_path, image_file)) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"File: {image_file}")
                
                img_path = os.path.join(yacht_path, image_file)
                img = tf.keras.preprocessing.image.load_img(img_path)
                x = tf.keras.preprocessing.image.img_to_array(img)
                x = x.reshape((1,) + x.shape)

                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_yacht_dir, save_prefix=yacht_name, save_format='jpeg'):
                    i += 1
                    if i > 5:  
                        break
