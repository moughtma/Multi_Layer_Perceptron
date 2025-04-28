import torch
import torch.nn as nn
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Create network architecture
class RiskClassifier(nn.Module):
    def __init__(self):
        super(RiskClassifier, self).__init__()
        # Input layer (4 features) → Hidden layer 1 (8 neurons)
        self.fc1 = nn.Linear(4, 8)
        # Hidden layer 1 → Hidden layer 2 (4 neurons)
        self.fc2 = nn.Linear(8, 4)
        # Hidden layer 2 → Output layer (1 neuron for binary classification)
        self.fc3 = nn.Linear(4, 1)
        # Using the relu activation function which works well for non-linear data
        self.relu = nn.ReLU()
        # Using the sigmoid activation which is ideal for binary classification and returns probability like output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # layer 1 calculations
        x = self.relu(self.fc1(x))
        # layer 2 calculations
        x = self.relu(self.fc2(x))
        # layer 3 calculations
        x = self.sigmoid(self.fc3(x))
        return x

# Step 2. Create some random weights and biases
# (In PyTorch, weights and biases are initialized automatically when you define layers)

# Step 3. Create training data and labels
# Inputs: [Latency, Traffic, Errors, Saturation]
def prepare_data():
    X = torch.tensor([
        # Synthetic training data in an array
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
    ], dtype=torch.float32)

    # Normalize features (mean 0, std 1) — column-wise
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    # Labels: 0 = OK, 1 = At risk
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32).unsqueeze(1)

    # Move data to device (CPU or GPU)
    return X.to(device), y.to(device)

# Step 4. Create the activation function
# (Handled inside the model with nn.ReLU() and nn.Sigmoid())

# Step 5. Create feed forward process
# (Handled inside the model's forward method)

# Step 6. Create the binary cross-entropy cost function
# (Using nn.BCELoss which handles it)

# Step 7. Create the backpropagation function
# (PyTorch automatically handles backpropagation via loss.backward())

# Step 8. Training loop using gradient descent
# X: Input data matrix (features), shape (10, 4)
# y: True labels, shape (10, 1)
# learning_rate: The learning rate for gradient descent updates
# epochs: Number of iterations to train
def train_model(model, X, y, learning_rate=0.001, epochs=2000):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        # Forward pass
        y_hat = model(X)

        # Calculate cost
        loss = criterion(y_hat, y)

        # Compute gradients via backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update weights and biases
        optimizer.step()

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            with torch.no_grad():
                predictions = (y_hat > 0.5).float()
                acc = (predictions == y).float().mean()
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.2f}")

# Step 9. Final prediction and accuracy
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_hat = model(X)
        predictions = (y_hat > 0.5).float()  # Threshold at 0.5 to round to 0 or 1
        accuracy = (predictions == y).float().mean()  # Compare predictions to actual labels
    print(f"Final accuracy: {accuracy.item():.2f}")
    return predictions, accuracy

# Load training data and labels
A0, Y = prepare_data()

# Create and move model to device
model = RiskClassifier().to(device)

# Train the model
train_model(model, A0, Y, learning_rate=0.001, epochs=2000)

# Evaluate the trained model
predictions, accuracy = evaluate_model(model, A0, Y)