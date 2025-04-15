
import numpy as np
# Step 1: Create network architecture
L = 3
n = [4, 8, 4, 1]

# Step 2. Create some random weights and biases

W1 = np.random.randn(n[1], n[0]) * np.sqrt(2. / n[0])
W2 = np.random.randn(n[2], n[1]) * np.sqrt(2. / n[1])
W3 = np.random.randn(n[3], n[2]) * np.sqrt(2. / n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

# Step 3. Create training data and labels
# Inputs: [Latency, Traffic, Errors, Saturation]
def prepare_data():
    X = np.array([
        #Synthetic training data in an array
        [100, 200, 1, 60],   # OK
        [300, 500, 4, 90],   # At risk
        [120, 180, 0, 50],   # OK
        [400, 600, 7, 95],   # At risk
        [150, 250, 2, 65],   # OK
        [250, 450, 5, 85],   # At risk
        [110, 190, 0, 55],   # OK
        [350, 550, 6, 92],   # At risk
        [130, 220, 1, 62],   # OK
        [280, 480, 4, 88]   # At risk
    ])
    # Normalize features (mean 0, std 1) — column-wise
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Labels: 0 = OK, 1 = At risk
    #The training data label with have dimensions of 1 x m.
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    m = X.shape[0]
    # print(X.shape)   prints (10, 4)
    # We need the matrix transposed as we will pass the data to the neural network in this rotation.
    # Use the .T property to do this.
    A0 = X.T
    # print(A0.shape)  prints (4, 10)
    #.reshape on a matrix allows for it to be reshaped. y is a vector,
    # but we want it as a matrix Y to match the output of our network.
    # print(y.shape) prints (10,)
    Y = y.reshape(n[L], m)
    # print(Y.shape) prints (1,10)
    return A0, Y

# Step 4. Create the activation function
# Using the relu activation function which works well for non-linear data
def relu(arr):
    return np.maximum(0, arr)
# Using the sigmoid activation which is ideal for binary classification and returns probability like output.
def sigmoid(arr):
  arr = np.clip(arr, -500, 500)  # Prevent overflow in exp() caused by very large numbers in the exponential.
  return 1 / (1 + np.exp(-1 * arr))

# Step 5. Create feed forward process
def feed_forward(A0):
    # layer 1 calculations
    # The shape of A0 is (4, 10).The shape of W1 is (8, 4).The shape of b1 is (8, 1).
    Z1 = W1 @ A0 + b1 # the @ symbol means matrix multiplication.
    # The shape of Z1 is (8, 10).
    A1 = relu(Z1)

    # layer 2 calculations
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    # layer 3 calculations
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # Cache the values which will be used for backpropagation
    cache = {
        "A0": A0,
        "A1": A1,
        "A2": A2
    }

    return A3, cache

# print(y_hat) outputs [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]] which means the model has predicted every input to be "at risk"
# The accuracy of the neural network is about 50%, no better than guessing with a coin toss.

# Step 6. Create the binary cross-entropy cost function
def cost_function(y_hat, Y):
    m = Y.shape[1]  # number of samples in the training data (columns in Y)
    epsilon = 1e-15  # prevents log(0) or log(1-1) which would result in infinity
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon) # ensure y_hat stays in reasonable range

    # This is the binary cross-entropy loss calculation with loss averaged over all samples
    cost = - (1 / m) * np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
    return cost

# print(cost) gives an output of 17.27 which is a high cost value. The cost should be close to 0 for a good model.
# The high cost value is due to the model predicting all inputs as "at risk" which is incorrect.

# Step 7. Create the backpropagation function
def backpropagation(y_hat, Y, cache):
    m = Y.shape[1]  # number of training examples in column vector Y. Need this for averaging the gradients.

    # Unpack cached activations
    A0 = cache["A0"] # input layer
    A1 = cache["A1"] # hidden layer 1 output after ReLU activation
    A2 = cache["A2"] # hidden layer 2 output after ReLU activation

    # Output layer (layer 3) gradients
    dZ3 = y_hat - Y # derivative of the loss function with respect to Z3. Shape: (1, 10)
    dW3 = (1 / m) * dZ3 @ A2.T           # Derivative of the loss function with respect to W3. Shape: (1, 4)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)  # Derivative of the loss function with respect to b3. Shape: (1, 1)

    # Layer 2 gradients
    dA2 = W3.T @ dZ3                      # Propagate the gradient back to the previous layer. Shape: (4, 10)
    dZ2 = dA2 * (A2 > 0)                  # ReLU derivative. Shape: (4, 10)
    dW2 = (1 / m) * dZ2 @ A1.T            # Derivative of the loss function with respect to W2. Shape: (4, 8)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Derivative of the loss function with respect to b2. Shape: (4, 1)

    # Layer 1 gradients
    dA1 = W2.T @ dZ2                      # Propagate the gradient back to the previous layer. Shape: (8, 10)
    dZ1 = dA1 * (A1 > 0)                  # ReLU derivative. Shape: (8, 10)
    dW1 = (1 / m) * dZ1 @ A0.T            # Derivative of the loss function with respect to W1. Shape: (8, 4)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Derivative of the loss function with respect to b1. Shape: (8, 1)

    # Return all gradients in a dictionary so they can be used later
    gradients = {
        "dW3": dW3, "db3": db3,
        "dW2": dW2, "db2": db2,
        "dW1": dW1, "db1": db1
    }

    return gradients

# Step 8. Training loop using gradient descent
# A0: Input data matrix (features), shape (4, m), Y: True labels, shape (1, m)
# learning_rate: The learning rate for gradient descent updates, epochs: Number of iterations to train
def train_model(A0, Y, learning_rate=0.001, epochs=2000):
    global W1, b1, W2, b2, W3, b3
    for epoch in range(epochs):
        # Forward pass
        A3, cache = feed_forward(A0)
        # Calculate cost
        cost = cost_function(A3, Y)
        # Compute gradients via backpropagation
        gradients = backpropagation(A3, Y, cache)

        # Update weights and biases
        W1 -= learning_rate * gradients["dW1"]
        b1 -= learning_rate * gradients["db1"]
        W2 -= learning_rate * gradients["dW2"]
        b2 -= learning_rate * gradients["db2"]
        W3 -= learning_rate * gradients["dW3"]
        b3 -= learning_rate * gradients["db3"]

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            predictions = (A3 > 0.5).astype(int)
            acc = np.mean(predictions == Y)
            print(f"Epoch {epoch}, Cost: {cost:.4f}, Accuracy: {acc:.2f}")
    #Returns updated global weights and biases (W1, b1, W2, b2, W3, b3)
    return W1, b1, W2, b2, W3, b3

# Step 9. Final prediction and accuracy
def evaluate_model(A0, Y):
    A3, _ = feed_forward(A0)
    predictions = (A3 > 0.5).astype(int) # This converts our sigmoid output to binary predictions by thresholding at 0.5 to round to 0 or 1
    accuracy = np.mean(predictions == Y) # Compare predictions to actual labels
    print("Final accuracy:", accuracy) # Display the accuracy of the model
    return predictions, accuracy

# Load training data and labels
A0, Y = prepare_data()

# Train the model
W1, b1, W2, b2, W3, b3 = train_model(A0, Y, learning_rate=0.001, epochs=2000)

# Evaluate the trained model
predictions, accuracy = evaluate_model(A0, Y)

import matplotlib.pyplot as plt


def train_model_with_cost(A0, Y, learning_rate=0.001, epochs=2000):
    """
    Trains the neural network and records the cost at each epoch.

    Parameters:
      A0: Input data matrix (features), shape (4, m)
      Y: True labels, shape (1, m)
      learning_rate: Learning rate for gradient descent updates.
      epochs: Number of training iterations.

    Returns:
      cost_history: A list of cost values recorded for each epoch.
    """
    global W1, b1, W2, b2, W3, b3
    cost_history = []

    for epoch in range(epochs):
        # Forward pass
        A3, cache = feed_forward(A0)
        # Compute cost and record it
        cost = cost_function(A3, Y)
        cost_history.append(cost)
        # Backpropagation to compute gradients
        gradients = backpropagation(A3, Y, cache)

        # Update weights and biases
        W1 -= learning_rate * gradients["dW1"]
        b1 -= learning_rate * gradients["db1"]
        W2 -= learning_rate * gradients["dW2"]
        b2 -= learning_rate * gradients["db2"]
        W3 -= learning_rate * gradients["dW3"]
        b3 -= learning_rate * gradients["db3"]

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            predictions = (A3 > 0.5).astype(int)
            acc = np.mean(predictions == Y)
            print(f"Epoch {epoch}, Cost: {cost:.4f}, Accuracy: {acc:.2f}")

    return cost_history


# Usage example:
# Prepare data
A0, Y = prepare_data()

# Train the model and record cost history
cost_history = train_model_with_cost(A0, Y, learning_rate=0.001, epochs=2000)

# Plotting the cost vs. iterations
plt.figure(figsize=(8, 6))
plt.plot(range(len(cost_history)), cost_history, label="Cost")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.legend()
plt.show()