import os
import numpy as np
import cv2
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import tensorflow as tf
from config.config import TRAINING_PATH, TESTING_PATH, IMAGE_SIZE, CLASS_NAMES

to_categorical = tf.keras.utils.to_categorical

def load_data():
    # 1)Load training data
    train_images, train_labels = [], []
    for class_name in CLASS_NAMES:
        class_path = os.path.join(TRAINING_PATH, class_name)
        for image_path in paths.list_images(class_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            train_images.append(image)
            train_labels.append(class_name)
    
    # 2)Load testing data
    test_images, test_labels = [], []
    for class_name in CLASS_NAMES:
        class_path = os.path.join(TESTING_PATH, class_name)
        for image_path in paths.list_images(class_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            test_images.append(image)
            test_labels.append(class_name)
    
    # 3)Convert to arrays and normalize
    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0
    
    # 4)One-hot encoding
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)  # Don't use to_categorical here
    test_labels = lb.transform(test_labels)       # LabelBinarizer already creates correct shape
    
    # 5)Shuffle training data
    train_images, train_labels = shuffle(train_images, train_labels)
    
    return train_images, test_images, train_labels, test_labels, lb