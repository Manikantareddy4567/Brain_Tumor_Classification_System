import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.data_preparation import load_data  # Fixed import
from config.config import BATCH_SIZE, CLASS_NAMES

load_model = tf.keras.models.load_model

def evaluate(model_path, lb=None):
    # 1)Load data and model
    _, test_X, _, test_Y, lb = load_data()  # Now properly imported
    model = load_model(model_path)
    
    # 2)Predict
    predictions = model.predict(test_X, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis=1)
    actuals = np.argmax(test_Y, axis=1)
    
    # 3)Classification report
    print("Classification Report:")
    print(classification_report(actuals, predictions, target_names=CLASS_NAMES))
    
    # 4)Confusion matrix
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 5)Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    # 6)Calculate accuracies
    accuracy = np.sum(actuals == predictions) / len(actuals)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\nPer-class Accuracy:")
    class_acc = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_idx = np.where(actuals == i)[0]
        class_correct = np.sum(predictions[class_idx] == actuals[class_idx])
        class_acc[class_name] = class_correct / len(class_idx)
        print(f"{class_name}: {class_acc[class_name]:.4f}")
    
    return accuracy, class_acc