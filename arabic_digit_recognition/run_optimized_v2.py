import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Import modules
from src.data_loader import ArabicDigitDataLoader

def main():
    # Create directories
    os.makedirs('models_optimized_v2', exist_ok=True)
    os.makedirs('results_optimized_v2', exist_ok=True)
    
    print("\n" + "="*60)
    print("OPTIMIZED V2 - TARGETING 99%+ ACCURACY (FIXED)")
    print("="*60)
    
    # Dataset path
    dataset_path = r"C:\Users\user\Downloads\Handwritten Arabic Numerals (0-9)\ANGKA ARAB"
    
    # Load data
    loader = ArabicDigitDataLoader(dataset_path)
    images, labels = loader.load_data(img_size=(64, 64))
    
    # Prepare data
    (X_train, y_train, y_train_labels), (X_val, y_val, y_val_labels), (X_test, y_test, y_test_labels) = loader.prepare_data()
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        shear_range=0.05,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Create enhanced model
    print("\nCreating enhanced CNN model...")
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(64,64,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.2),
        
        # Block 2
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # FIXED: Use simple learning rate with decay instead of schedule
    initial_lr = 0.001
    decay_steps = 30 * (len(X_train) // 32)
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=initial_lr,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # FIXED: Added learning rate decay callback
    def lr_scheduler(epoch, lr):
        if epoch < 15:
            return lr
        elif epoch < 25:
            return lr * 0.5
        elif epoch < 35:
            return lr * 0.25
        else:
            return lr * 0.1
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models_optimized_v2/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)  # NEW
    ]
    
    print("\nStarting optimized V2 training...")
    print("Target: 99%+ accuracy")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=45,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=len(X_train) // 32
    )
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate per-class accuracy
    print(f"\n{'='*60}")
    print(f"🎯 FINAL TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\n📊 Per-Class Accuracy:")
    print("-" * 40)
    
    for i in range(10):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
            emoji = "🏆" if class_acc == 1.0 else "📈" if class_acc > 0.98 else "🔧"
            print(f"Digit {i}: {class_acc:.4f} ({class_acc*100:.2f}%) {emoji}")
    
    # Save model
    model.save('models_optimized_v2/final_model.keras')
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy - V2', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.9, 1.0])
    
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss - V2', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_optimized_v2/training_history.png', dpi=300)
    plt.close()
    
    # Save summary with comparison
    with open('results_optimized_v2/summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("OPTIMIZED V2 RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Previous Best: 0.9886 (98.86%)\n")
        f.write(f"Improvement: {(test_accuracy - 0.9886)*100:+.2f}%\n\n")
        f.write("Per-Class Accuracy:\n")
        for i in range(10):
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_acc = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
                f.write(f"Digit {i}: {class_acc:.4f}\n")
    
    print(f"\n Results saved in 'results_optimized_v2' folder")
    print(f" Model saved in 'models_optimized_v2' folder")
    
    # Show comparison
    print(f"\n Comparison with previous run:")
    print(f"   Previous: 98.86%")
    print(f"   Current:  {test_accuracy*100:.2f}%")
    print(f"   Change:   {(test_accuracy - 0.9886)*100:+.2f}%")

if __name__ == "__main__":
    main()