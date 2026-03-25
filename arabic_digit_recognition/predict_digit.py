import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load your trained model
def load_best_model():
    model_path = 'models_optimized_v2/best_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f" Model loaded from {model_path}")
        return model
    else:
        print(f" Model not found at {model_path}")
        return None

# Preprocess a single image
def preprocess_image(image_path, img_size=(64, 64)):
    """Load and preprocess a single image for prediction"""
    
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f" Could not read image: {image_path}")
        return None
    
    # Resize
    img = cv2.resize(img, img_size)
    
    # Apply same preprocessing as training
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Normalize and reshape for model
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 64, 64, 1)  # Add batch dimension
    
    return img, cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), img_size)

# Predict digit from image
def predict_digit(model, image_path):
    """Predict digit from an image file"""
    
    # Preprocess image
    result = preprocess_image(image_path)
    if result is None:
        return None
    
    processed_img, original_img = result
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)[0]
    predicted_digit = np.argmax(predictions)
    confidence = predictions[predicted_digit] * 100
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3_confidences = [predictions[i] * 100 for i in top_3_indices]
    
    return {
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'top_3': list(zip(top_3_indices, top_3_confidences)),
        'all_probabilities': predictions,
        'original_img': original_img,
        'processed_img': processed_img[0, :, :, 0]
    }

# Display prediction result
def show_prediction(result, image_path):
    """Display the image with prediction"""
    
    if result is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(result['original_img'], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Processed image (what model sees)
    axes[1].imshow(result['processed_img'], cmap='gray')
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    
    # Prediction bar chart
    digits = range(10)
    colors = ['green' if d == result['predicted_digit'] else 'gray' for d in digits]
    bars = axes[2].bar(digits, result['all_probabilities'] * 100, color=colors)
    axes[2].set_xlabel('Digit')
    axes[2].set_ylabel('Confidence (%)')
    axes[2].set_title('Prediction Probabilities')
    axes[2].set_ylim([0, 100])
    
    # Add confidence values on bars
    for bar, prob in zip(bars, result['all_probabilities'] * 100):
        height = bar.get_height()
        if height > 5:  # Only show if >5%
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f" Predicted Digit: {result['predicted_digit']} (Confidence: {result['confidence']:.2f}%)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Predict multiple images from a folder
def predict_folder(model, folder_path):
    """Predict all images in a folder"""
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f" No images found in {folder_path}")
        return
    
    print(f"\n Found {len(image_files)} images in folder")
    print("="*50)
    
    for img_file in image_files[:5]:  # Show first 5 only
        img_path = os.path.join(folder_path, img_file)
        result = predict_digit(model, img_path)
        
        if result:
            print(f"\n📷 {img_file}:")
            print(f"   Predicted: {result['predicted_digit']} (Confidence: {result['confidence']:.2f}%)")
            
            # Show top 3 predictions
            top_3_str = ", ".join([f"{d}({c:.1f}%)" for d, c in result['top_3']])
            print(f"   Top 3: {top_3_str}")

# Interactive prediction
def interactive_mode(model):
    """Let user input image paths interactively"""
    
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter image paths (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*40)
        image_path = input(" Enter image path: ").strip()
        
        if image_path.lower() in ['quit', 'q', 'exit']:
            break
        
        if not os.path.exists(image_path):
            print(f" File not found: {image_path}")
            continue
        
        result = predict_digit(model, image_path)
        if result:
            show_prediction(result, image_path)
            
            # Also print text result
            print(f"\n Prediction: Digit {result['predicted_digit']}")
            print(f"   Confidence: {result['confidence']:.2f}%")
            print(f"   Top 3: {result['top_3']}")

def main():
    print("\n" + "="*60)
    print(" ARABIC DIGIT PREDICTION SYSTEM")
    print("="*60)
    print(f"Using model: 99.14% accuracy")
    
    # Load model
    model = load_best_model()
    if model is None:
        return
    
    print("\n Model ready for predictions!")
    
    # Menu
    while True:
        print("\n" + "="*40)
        print("OPTIONS:")
        print("1. Predict single image")
        print("2. Predict all images in a folder")
        print("3. Interactive mode (show visualizations)")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                result = predict_digit(model, image_path)
                if result:
                    print(f"\n Predicted: {result['predicted_digit']} (Confidence: {result['confidence']:.2f}%)")
                    print(f"   Top 3: {result['top_3']}")
            else:
                print(" File not found")
        
        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()
            predict_folder(model, folder_path)
        
        elif choice == '3':
            interactive_mode(model)
        
        elif choice == '4':
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice")

if __name__ == "__main__":
    main()