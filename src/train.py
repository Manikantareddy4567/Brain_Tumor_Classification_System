import os
import tensorflow as tf
from src.model import build_model
from src.data_preparation import load_data
from config.config import BATCH_SIZE, EPOCHS

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

def train():
    # 1)Load data
    train_X, test_X, train_Y, test_Y, lb = load_data()
    
    # 2)Add these verification lines:
    print(f"Train X shape: {train_X.shape}")
    print(f"Train Y shape: {train_Y.shape}")
    print(f"Test X shape: {test_X.shape}")
    print(f"Test Y shape: {test_Y.shape}")
    
    # 3)Data augmentation
    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # 4)Build model
    model = build_model()
    
    # 5)Calculate steps
    train_steps = len(train_X) // BATCH_SIZE
    
    # 6)rain model
    history = model.fit(
        train_generator.flow(train_X, train_Y, batch_size=BATCH_SIZE),
        steps_per_epoch=train_steps,
        validation_data=(test_X, test_Y),
        epochs=EPOCHS
    )
    
    # 7)Save model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "brain_tumor_model.h5"))
    
    return history, lb