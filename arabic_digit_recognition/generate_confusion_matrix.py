import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import data loader
from src.data_loader import ArabicDigitDataLoader

def main():
    print("\n" + "="*60)
    print(" GENERATING CONFUSION MATRIX")
    print("="*60)
    
    # Load the trained model
    model_path = 'models_optimized_v2/best_model.keras'
    if not os.path.exists(model_path):
        print(f" Model not found at {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f" Model loaded from {model_path}")
    
    # Load test data
    dataset_path = r"C:\Users\user\Downloads\Handwritten Arabic Numerals (0-9)\ANGKA ARAB"
    loader = ArabicDigitDataLoader(dataset_path)
    images, labels = loader.load_data(img_size=(64, 64))
    
    # Get test data
    (_, _, _), (_, _, _), (X_test, y_test, y_test_labels) = loader.prepare_data()
    
    # Make predictions
    print("\n🔍 Making predictions on test set...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\n Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                annot_kws={'size': 14})
    plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    # Save to results_optimized_v2 folder
    output_path = 'results_optimized_v2/confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved to: {output_path}")
    plt.close()
    
    # Also save a text version with per-class metrics
    print("\n Classification Report:")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)])
    print(report)
    
    # Save report to file
    report_path = 'results_optimized_v2/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # Calculate per-digit accuracy
    print("\n Per-Digit Accuracy:")
    print("-" * 40)
    for i in range(10):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            digit_acc = np.sum(y_pred[mask] == i) / np.sum(mask)
            print(f"Digit {i}: {digit_acc*100:.2f}%")
    
    print(f"\n All files saved in 'results_optimized_v2' folder:")
    print(f"   - confusion_matrix.png")
    print(f"   - classification_report.txt")

if __name__ == "__main__":
    main()