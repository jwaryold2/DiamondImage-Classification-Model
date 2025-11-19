import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

# Specify the path to your dataset
train_data_dir = 'DiamondImages/train'
test_data_dir = 'DiamondImages/test'

# Define data generators for training and testing with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generators with 224x224 target size
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(112, 112),  # Updated target size
    batch_size=15,
    class_mode='categorical',
    classes=['CUSHION', 'EMERALD', 'HEART', 'OVAL', 'ROUND', 'RADIANT']
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(112, 112),  # Updated target size
    batch_size=15,
    class_mode='categorical',
    classes=['CUSHION', 'EMERALD', 'HEART', 'OVAL', 'ROUND', 'RADIANT']
)

# Use a more complex model suitable for multi-label classification
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='sigmoid'))  # Adjust units for the number of classes

# Compile the model for multi-label classification
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Change the loss function
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train the model with augmented data
model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[tensorboard_callback])
