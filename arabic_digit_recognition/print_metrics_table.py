import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from src.data_loader import ArabicDigitDataLoader

def print_fancy_table():
    print("\n" + "="*80)
    print("HANDWRITTEN ARABIC DIGIT RECOGNITION - PERFORMANCE REPORT")
    print("="*80)
    
    # Load model and data
    print("\nLoading model and test data...")
    model = tf.keras.models.load_model('models_optimized_v2/best_model.keras')
    
    dataset_path = r"C:\Users\user\Downloads\Handwritten Arabic Numerals (0-9)\ANGKA ARAB"
    loader = ArabicDigitDataLoader(dataset_path)
    images, labels = loader.load_data(img_size=(64, 64))
    (_, _, _), (_, _, _), (X_test, y_test, y_test_labels) = loader.prepare_data()
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Calculate support (number of samples per class)
    support = [np.sum(y_true == i) for i in range(10)]
    
    # Print header
    print("\n" + "*"*80)
    print("*" + " "*30 + "MODEL PERFORMANCE SUMMARY" + " "*30 + "*")
    print("*"*80)
    
    print(f"\nOVERALL METRICS:")
    print(f"   Test Accuracy:  {accuracy*100:.2f}%")
    print(f"   Macro Precision: {precision_macro*100:.2f}%")
    print(f"   Macro Recall:    {recall_macro*100:.2f}%")
    print(f"   Macro F1-Score:  {f1_macro*100:.2f}%")
    
    # Print detailed table
    print("\n" + "-"*80)
    print("PER-CLASS PERFORMANCE BREAKDOWN")
    print("-"*80)
    
    # Table header
    print("\n" + "─"*80)
    print(f"{'Digit':^10} | {'Precision':^12} | {'Recall':^12} | {'F1-Score':^12} | {'Support':^10} | {'Performance':^12}")
    print("─"*80)
    
    # Table rows
    for i in range(10):
        # Determine performance category
        if f1_per_class[i] == 1.0:
            status = "PERFECT"
        elif f1_per_class[i] >= 0.99:
            status = "EXCELLENT"
        elif f1_per_class[i] >= 0.98:
            status = "GOOD"
        elif f1_per_class[i] >= 0.97:
            status = "FAIR"
        else:
            status = "NEEDS WORK"
        
        print(f"{i:^10} | {precision_per_class[i]*100:^11.2f}% | {recall_per_class[i]*100:^11.2f}% | "
              f"{f1_per_class[i]*100:^11.2f}% | {support[i]:^10} | {status:^12}")
    
    print("─"*80)
    
    # Find best and worst digits
    best_digit = np.argmax(f1_per_class)
    worst_digit = np.argmin(f1_per_class)
    
    print("\n" + "*"*80)
    print("HIGHLIGHTS")
    print("*"*80)
    
    print(f"\nBEST PERFORMING DIGIT:  Digit {best_digit}")
    print(f"   Precision: {precision_per_class[best_digit]*100:.2f}%")
    print(f"   Recall:    {recall_per_class[best_digit]*100:.2f}%")
    print(f"   F1-Score:  {f1_per_class[best_digit]*100:.2f}%")
    
    print(f"\nDIGIT NEEDING IMPROVEMENT:  Digit {worst_digit}")
    print(f"   Precision: {precision_per_class[worst_digit]*100:.2f}%")
    print(f"   Recall:    {recall_per_class[worst_digit]*100:.2f}%")
    print(f"   F1-Score:  {f1_per_class[worst_digit]*100:.2f}%")
    
    # Perfect digits count
    perfect_count = np.sum(f1_per_class == 1.0)
    print(f"\nPERFECT DIGITS (100% F1-Score): {perfect_count}/10")
    perfect_digits = [i for i in range(10) if f1_per_class[i] == 1.0]
    if perfect_digits:
        print(f"   Digits: {perfect_digits}")
    
    # Confusion pairs
    print("\n" + "-"*80)
    print("TOP MISCLASSIFICATIONS")
    print("-"*80)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find top 3 confusions (off-diagonal)
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], i, j))
    
    confusions.sort(reverse=True)
    
    print("\nMost common misclassifications:")
    for count, true, pred in confusions[:3]:
        percentage = (count / support[true]) * 100
        print(f"   Digit {true} → {pred}: {count} times ({percentage:.1f}% of digit {true})")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL MODEL ASSESSMENT")
    print("="*80)
    print(f"   Overall Accuracy: {accuracy*100:.2f}%")
    print(f"   Perfect Digits: {perfect_count}/10")
    print(f"   Macro F1-Score: {f1_macro*100:.2f}%")
    print("="*80)

if __name__ == "__main__":
    print_fancy_table()