import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Paths
train_dir = r'C:\Users\USER\Desktop\HAR_XGBoost project\Human Action Recognition\train'
csv_file = r'C:\Users\USER\Desktop\HAR_XGBoost project\Human Action Recognition\Training_set.csv'
model_save_path = 'cnn_har_model.keras'
label_encoder_path = 'label_encoder.pkl'

# Load CSV with image paths and labels
data = pd.read_csv(csv_file)

# Encode labels
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['label'])

# Save the label encoder
joblib.dump(label_encoder, label_encoder_path)

# Split data into training and validation sets
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# Image data generator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load data generators
def load_data(dataframe, datagen, batch_size, img_size=(224, 224)):
    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=train_dir,
        x_col='filename',
        y_col='encoded_label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    return generator

train_generator = load_data(train_df, train_datagen, batch_size=32)
val_generator = load_data(val_df, val_datagen, batch_size=32)

# Define MobileNetV2-based CNN
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(label_encoder.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')
]

# Train the model
model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Evaluation
val_loss, val_acc = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Optional Fine-tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save final model again (overwrites best checkpoint if needed)
model.save(model_save_path)
