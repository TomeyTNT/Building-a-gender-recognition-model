import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(rescale=1./255,)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/ahmadi/yolo_labeled_resized_split/train',
    target_size=(128, 128),
    batch_size=1,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'C:/Users/ahmadi/yolo_labeled_resized_split/val',
    target_size=(128, 128),
    batch_size=1,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/ahmadi/yolo_labeled_resized_split/test',
    target_size=(128, 128),
    batch_size=1,
    class_mode='binary'
)

class_weights = {0: 1.3, 1: 4}
callback = callbacks.EarlyStopping(monitor='val_loss', min_delta= 1e-3, patience=5, verbose=1, restore_best_weights = True)

model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    # verbose=1,
    class_weight=class_weights,
    callbacks=[callback]
)

# ارزیابی مدل
train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Train accuracy: {train_accuracy}')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

