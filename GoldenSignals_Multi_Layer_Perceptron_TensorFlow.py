import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# Step 1: Create network architecture
L = 3
n = [4, 8, 4, 1]  # 4 input features, 8 neurons in layer 1, 4 in layer 2, 1 output

# Step 2. No need to manually create weights and biases in Keras
# Step 3. Create training data and labels
# Inputs: [Latency, Traffic, Errors, Saturation]
def prepare_data():
    X = np.array([
        [100, 200, 1, 60],   # OK
        [300, 500, 4, 90],   # At risk
        [120, 180, 0, 50],   # OK
        [400, 600, 7, 95],   # At risk
        [150, 250, 2, 65],   # OK
        [250, 450, 5, 85],   # At risk
        [110, 190, 0, 55],   # OK
        [350, 550, 6, 92],   # At risk
        [130, 220, 1, 62],   # OK
        [280, 480, 4, 88]    # At risk
    ])
    # Normalize features (mean 0, std 1)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Labels: 0 = OK, 1 = At risk
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y

# Step 4. Activation functions are handled internally by Keras
# Step 5. Feed forward process is handled by Keras when calling model.fit()
# Step 6. Binary cross-entropy cost function is built-in
# Step 7. Backpropagation is handled by Keras
# Step 8. Training loop using gradient descent (Keras handles this)
def train_model(X, y, learning_rate=0.001, epochs=2000): # Train the model learning rate and epochs
    model = models.Sequential([ # Create a sequential model
        layers.Input(shape=(n[0],)),              # Input layer (4 features)
        layers.Dense(n[1], activation='relu'),    # Layer 1 using ReLU
        layers.Dense(n[2], activation='relu'),    # Layer 2 using ReLU
        layers.Dense(n[3], activation='sigmoid')  # Output layer using Sigmoid
    ])

    model.compile( # Compile the model
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), # Adaptive Moment Estimation (Adam) optimizer helps to speed up learning
        #by remembering past gradients and adapts the learning rate for each parameter based on recent gradients (first and second moments).
        loss='binary_crossentropy', # Binary cross-entropy loss for binary classification
        metrics=['accuracy'] # Track and report accuracy during training and evaluation
    )

    model.fit(X, y, epochs=epochs, verbose=0) # Train the model without verbose output
    return model

# Step 9. Final prediction and accuracy
def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y, verbose=0) # loss (how wrong the predictions are) and accuracy (how many correct predictions)
    print("Final accuracy:", accuracy) # Print final accuracy
    predictions = (model.predict(X) > 0.5).astype(int) # Predict and convert probabilities to binary (0 or 1). asType(int) converts the boolean array to an integer array.
    return predictions, accuracy # Return predictions and accuracy

# Load training data and labels
A0, Y = prepare_data()

# Train the model
model = train_model(A0, Y, learning_rate=0.001, epochs=2000)

# Evaluate the trained model
predictions, accuracy = evaluate_model(model, A0, Y)

# Display predictions
print("Predictions:", predictions.flatten())