import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import os
import numpy as np

def create_data_generators(train_dir, validation_dir, test_dir, batch_size=32, img_size=(224, 224)):
    """
    Create data pipelines with CORRECT label mapping
    """
    print("Creating datasets with correct label mapping...")
    
    # Create datasets
    train_dataset = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True,
        seed=42
    )
    
    validation_dataset = image_dataset_from_directory(
        validation_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    test_dataset = image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    # Print class mapping for verification
    print(f"Class names: {train_dataset.class_names}")
    print(f"Class indices: {dict(zip(train_dataset.class_names, range(len(train_dataset.class_names))))}")
    print("NOTE: In binary classification, class 0 = 0.0, class 1 = 1.0")
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    # Preprocessing functions
    def preprocess_train(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = data_augmentation(image)
        return image, label
    
    def preprocess_test(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_train)
    validation_dataset = validation_dataset.map(preprocess_test)
    test_dataset = test_dataset.map(preprocess_test)
    
    # Optimize performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset, test_dataset