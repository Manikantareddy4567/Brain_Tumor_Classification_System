import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.train import train
from src.evaluate import evaluate

if __name__ == "__main__":
    # 1)Train model
    history, label_binarizer = train()
    
    # 2)Evaluate model
    model_path = os.path.join(os.path.dirname(__file__), "model", "brain_tumor_model.h5")
    evaluate(model_path, label_binarizer)