import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class ArabicDigitDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = []
        self.labels = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
    def load_data(self, img_size=(64, 64)):
        print("Loading dataset...")
        
        if not os.path.exists(self.data_path):
            print(f"Path does not exist: {self.data_path}")
            return None, None
        
        digit_folders = [f for f in os.listdir(self.data_path) 
                        if os.path.isdir(os.path.join(self.data_path, f)) and f.isdigit()]
        digit_folders.sort()
        
        print(f"Found {len(digit_folders)} digit folders")
        
        for digit_folder in digit_folders:
            digit = int(digit_folder)
            folder_path = os.path.join(self.data_path, digit_folder)
            
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading digit {digit}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = self.preprocess_image(img)
                    self.images.append(img)
                    self.labels.append(digit)
        
        if len(self.images) == 0:
            print("No images loaded!")
            return None, None
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\nTotal images loaded: {len(self.images)}")
        return self.images, self.labels
    
    def preprocess_image(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = img.astype('float32') / 255.0
        return img
    
    def prepare_data(self, test_size=0.15, validation_size=0.15):
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels, test_size=test_size, 
            random_state=42, stratify=self.labels
        )
        
        val_relative_size = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, 
            random_state=42, stratify=y_temp
        )
        
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
        X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], 1)
        
        y_train_cat = to_categorical(y_train, 10)
        y_val_cat = to_categorical(y_val, 10)
        y_test_cat = to_categorical(y_test, 10)
        
        print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        return (X_train, y_train_cat, y_train), (X_val, y_val_cat, y_val), (X_test, y_test_cat, y_test)
    
    def visualize_samples(self, num_samples_per_class=2):
        fig, axes = plt.subplots(10, num_samples_per_class, figsize=(12, 20))
        
        for digit in range(10):
            digit_indices = np.where(self.labels == digit)[0]
            for j in range(min(num_samples_per_class, len(digit_indices))):
                idx = digit_indices[j]
                img = self.images[idx]
                axes[digit, j].imshow(img, cmap='gray')
                axes[digit, j].set_title(f'Digit: {digit}')
                axes[digit, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_digits.png', dpi=300, bbox_inches='tight')
        plt.show()
