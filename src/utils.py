import matplotlib.pyplot as plt
import numpy as np

def plot_image(image_array):
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()
    
def plot_sample(images, labels, lb, n=5):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        idx = np.random.randint(0, len(images))
        plt.subplot(1, n, i+1)
        plt.imshow(images[idx])
        plt.title(lb.classes_[np.argmax(labels[idx])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()