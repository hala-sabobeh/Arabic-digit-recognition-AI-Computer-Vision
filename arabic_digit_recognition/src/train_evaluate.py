import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.history = None
        
    def compile_model(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled")
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.history
    
    def plot_training_history(self):
        if self.history is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history.history['accuracy'], label='Training')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['loss'], label='Training')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, X_test, y_test, y_test_labels, class_names):
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred, target_names=class_names))
        
        cm = confusion_matrix(y_test_labels, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return test_accuracy, cm
    
    def visualize_predictions(self, X_test, y_test_labels, num_samples=20):
        y_pred_probs = self.model.predict(X_test[:num_samples], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        n_cols = 5
        n_rows = (num_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(X_test[i].reshape(64, 64), cmap='gray')
            color = 'green' if y_pred[i] == y_test_labels[i] else 'red'
            axes[i].set_title(f'True:{y_test_labels[i]}, Pred:{y_pred[i]}', color=color, fontsize=8)
            axes[i].axis('off')
        
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
