import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

train_data_dir = 'Diamond/Images/train'
test_data_dir = 'Diamond/Images/test'

# ---- DATA AUGMENTATION ----
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

# ---- DATA GENERATORS ----
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(300, 300),
    batch_size=15,
    class_mode='categorical',
    classes=['CUSHION', 'EMERALD', 'HEART', 'OVAL', 'ROUND', 'RADIANT', 'NONE']
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(300, 300),
    batch_size=15,
    class_mode='categorical',
    classes=['CUSHION', 'EMERALD', 'HEART', 'OVAL', 'ROUND', 'RADIANT', 'NONE']
)

# ---- MODEL ----
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# ---- TRAINING ----
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

test_images, test_labels = next(test_generator)

predictions = model.predict(test_images)

class_names = ['CUSHION', 'EMERALD', 'HEART', 'OVAL', 'ROUND', 'RADIANT', 'NONE']

plt.figure(figsize=(15, 15))

for i in range(9):  # show 9 predictions
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])

    pred_class = class_names[np.argmax(predictions[i])]
    true_class = class_names[np.argmax(test_labels[i])]

    plt.title(f"Predicted: {pred_class}\nActual: {true_class}")
    plt.axis('off')

plt.tight_layout()
plt.show()
