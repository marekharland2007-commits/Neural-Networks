import numpy as np
import gzip
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z):
    """Derivative of sigmoid: σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(z):
    "Derivative of ReLU, returns 1 for z > 0 and 0 otherwise"
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def load_mnist():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

def init_network_architecture(input_size, hidden_size1, hidden_size2, output_size):
    """Initialize weights and biases for a 3-layer neural network using He initialization for ReLU activations."""
    W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size1))

    W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
    b2 = np.zeros((1, hidden_size2))

    W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
    b3 = np.zeros((1, output_size))

    return W1, b1, W2, b2, W3, b3

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

def backward_propagation(X, y_true, z1, a1, z2, a2, z3, a3, W2, W3):
    m = y_true.shape[0] 

    dz3 = (a3 - y_true)
    dW3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m

    da2 = np.dot(dz3, W3.T)
    dz2 = da2 * relu_derivative(z2)

    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    return W1, b1, W2, b2, W3, b3

def train(X_train, y_train, X_valid, y_valid, W1, b1, W2, b2, W3, b3, initial_learning_rate, epochs):
    best_val_accuracy = 0.0
    best_W1, best_b1, best_W2, best_b2, best_W3, best_b3 = W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy()
    noImprovement_epochs = 0
    learning_rate = initial_learning_rate
    epoch_reductions = 0

    for epoch in range(epochs):

        if epoch == (epochs // 4) and epoch_reductions == 0:
            learning_rate *= 0.75
            print(f"Learning rate reduced to {learning_rate} at epoch {epoch+1}")
            epoch_reductions += 1
            W1, b1, W2, b2, W3, b3 = best_W1.copy(), best_b1.copy(), best_W2.copy(), best_b2.copy(), best_W3.copy(), best_b3.copy()
        
        if epoch == (epochs // 2) and epoch_reductions <= 1:
            learning_rate *= 0.75
            print(f"Learning rate reduced to {learning_rate} at epoch {epoch+1}")
            epoch_reductions += 1
            W1, b1, W2, b2, W3, b3 = best_W1.copy(), best_b1.copy(), best_W2.copy(), best_b2.copy(), best_W3.copy(), best_b3.copy()

        elif epoch == (epochs // 4 * 3) and epoch_reductions <= 2:
            learning_rate *= 0.75
            print(f"Learning rate reduced to {learning_rate} at epoch {epoch+1}")
            epoch_reductions += 1
            W1, b1, W2, b2, W3, b3 = best_W1.copy(), best_b1.copy(), best_W2.copy(), best_b2.copy(), best_W3.copy(), best_b3.copy()



        z1, a1, z2, a2, z3, a3 = forward_propagation(X_train, W1, b1, W2, b2, W3, b3)
        loss = compute_loss(y_train, a3)
        accuracy = evaluate(X_train, y_train, W1, b1, W2, b2, W3, b3)

        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X_train, y_train, z1, a1, z2, a2, z3, a3, W2, W3)

        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1,
                                                   dW2, db2, dW3, db3, learning_rate)

        val_accuracy = evaluate(X_valid, y_valid, W1, b1, W2, b2, W3, b3)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy
            best_W1, best_b1, best_W2, best_b2, best_W3, best_b3 = W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy()
            noImprovement_epochs = 0
        else:
            noImprovement_epochs += 1

    return best_W1, best_b1, best_W2, best_b2, best_W3, best_b3

def evaluate(X, y, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, a3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(a3, axis=1)

    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    return accuracy

def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoded format"""
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


if __name__ == "__main__": 
    train_set, valid_set, test_set = load_mnist()
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    # One-hot encode the labels
    y_train_encoded = one_hot_encode(y_train)
    y_valid_encoded = one_hot_encode(y_valid)
    y_test_encoded = one_hot_encode(y_test)

    input_size = X_train.shape[1]
    output_size = 10

    W1, b1, W2, b2, W3, b3 = init_network_architecture(input_size, 256, 128, output_size)
    learning_rate = 0.75
    epochs = 300

    W1, b1, W2, b2, W3, b3 = train(X_train, y_train_encoded, X_valid, y_valid_encoded, W1, b1, W2, b2, W3, b3, learning_rate, epochs)
    train_accuracy = evaluate(X_train, y_train_encoded, W1, b1, W2, b2, W3, b3)
    valid_accuracy = evaluate(X_valid, y_valid_encoded, W1, b1, W2, b2, W3, b3)
    test_accuracy = evaluate(X_test, y_test_encoded, W1, b1, W2, b2, W3, b3)
    print(f'Final Testing Accuracy: {test_accuracy:.4f}')

    weights = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2,
    "W3": W3,
    "b3": b3
    }

    with open("digit_model.pkl", "wb") as f:
        pickle.dump(weights, f)
