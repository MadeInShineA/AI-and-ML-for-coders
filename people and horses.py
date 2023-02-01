import urllib.request
import zipfile
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = 'horse-or-human/training/'
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory( training_dir, target_size=(300, 300), class_mode='binary')


