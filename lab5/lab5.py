import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
from typing import List, Callable
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Seed for reproducibility
np.random.seed(42)

# 1. CORE LAYER CLASSES


class Layer(ABC):
    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert 0 < learning_rate < 1
        self._learning_rate = learning_rate

class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # He initialization for ReLU activation
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        input_error_derivative = np.dot(output_error_derivative, self.weights.T)
        weights_derivative = np.dot(self.input.T, output_error_derivative)
        bias_derivative = np.sum(output_error_derivative, axis=0, keepdims=True)

        # SGD parameter update
        self.weights -= self.learning_rate * weights_derivative
        self.bias -= self.learning_rate * bias_derivative

        return input_error_derivative

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        # Derivative is 1 if x > 0, else 0
        return output_error_derivative * (self.input > 0)


# 2. GROUP D - LOSS FUNCTIONS


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    n = y_pred.shape[0] if len(y_pred.shape) > 1 else y_pred.size
    return np.sum(np.power(y_pred - y_true, 2)) / n

def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = y_pred.shape[0] if len(y_pred.shape) > 1 else y_pred.size
    return 2 * (y_pred - y_true) / n

def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    n = y_pred.shape[0] if len(y_pred.shape) > 1 else y_pred.size
    return np.sum(np.abs(y_pred - y_true)) / n

def mae_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = y_pred.shape[0] if len(y_pred.shape) > 1 else y_pred.size
    return np.sign(y_pred - y_true) / n

def softmax(x: np.ndarray) -> np.ndarray:
    # Shift values for numerical stability
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    probs = softmax(y_pred)
    # Epsilon clip to avoid log(0)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(probs)) / y_pred.shape[0]

def cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Combined softmax and CE derivative
    return (softmax(y_pred) - y_true) / y_pred.shape[0]

class Loss:
    def __init__(self, loss_function: Callable, loss_function_derivative: Callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_function(y_pred, y_true)

    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.loss_function_derivative(y_pred, y_true)



# 3. NETWORK CLASS


class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = None

    def compile(self, loss: Loss) -> None:
        self.loss = loss
        for layer in self.layers:
            layer.learning_rate = self.learning_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, verbose: int = 0) -> List[float]:
        self.learning_rate = learning_rate
        for layer in self.layers:
            layer.learning_rate = learning_rate

        loss_history = []
        for epoch in range(epochs):
            y_pred = self(x_train)
            epoch_loss = self.loss.loss(y_pred, y_train)
            loss_history.append(epoch_loss)
            
            error_derivative = self.loss.loss_derivative(y_pred, y_train)
            for layer in reversed(self.layers):
                error_derivative = layer.backward(error_derivative)
                
            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")
                
        return loss_history



# 4. EXPERIMENTS AND PLOTTING


def generate_loss_curves(X_train, y_train):
    print("\n--- Generating loss_curves.png ---")
    epochs, lr = 20, 0.05
    X_sub, y_sub = X_train[:3000], y_train[:3000] # Subset for faster training

    configs = [
        ("Cross-Entropy", cross_entropy, cross_entropy_derivative),
        ("MSE", mse, mse_derivative),
        ("MAE", mae, mae_derivative)
    ]
    
    plt.figure(figsize=(10, 6))
    
    for name, loss_fn, loss_grad in configs:
        print(f"Training with {name}...")
        net = Network([FullyConnected(784, 64), ReLU(), FullyConnected(64, 10)], lr)
        net.compile(Loss(loss_fn, loss_grad))
        history = net.fit(X_sub, y_sub, epochs=epochs, learning_rate=lr, verbose=0)
        plt.plot(history, label=f'{name} Loss', linewidth=2)

    plt.title('Training Loss Curves by Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()
    print("=> Saved 'loss_curves.png'")

def generate_arch_comparison(X_train, y_train):
    print("\n--- Generating arch_comparison.png ---")
    epochs, lr = 20, 0.05
    X_sub, y_sub = X_train[:3000], y_train[:3000]

    print("Training Shallow Architecture...")
    net_shallow = Network([FullyConnected(784, 32), ReLU(), FullyConnected(32, 10)], lr)
    net_shallow.compile(Loss(cross_entropy, cross_entropy_derivative))
    hist_shallow = net_shallow.fit(X_sub, y_sub, epochs=epochs, learning_rate=lr, verbose=0)

    print("Training Deep Architecture...")
    net_deep = Network([
        FullyConnected(784, 128), ReLU(), 
        FullyConnected(128, 64), ReLU(), 
        FullyConnected(64, 32), ReLU(), 
        FullyConnected(32, 10)
    ], lr)
    net_deep.compile(Loss(cross_entropy, cross_entropy_derivative))
    hist_deep = net_deep.fit(X_sub, y_sub, epochs=epochs, learning_rate=lr, verbose=0)

    plt.figure(figsize=(10, 6))
    plt.plot(hist_shallow, label='Shallow (1 Hidden Layer)', linewidth=2, linestyle='--')
    plt.plot(hist_deep, label='Deep (3 Hidden Layers)', linewidth=2)
    plt.title('Cross-Entropy Loss: Shallow vs Deep Architecture')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('arch_comparison.png')
    plt.close()
    print("=> Saved 'arch_comparison.png'")



# 5. MAIN EXECUTION


if __name__ == "__main__":
    print("Loading MNIST dataset... (This might take a minute)")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
    
    # Normalize features and one-hot encode targets
    X = mnist.data / 255.0
    y_onehot = np.eye(10)[mnist.target.astype(int)]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
    print("Data loaded. Starting experiments...")
    
    generate_loss_curves(X_train, y_train)
    generate_arch_comparison(X_train, y_train)
    
    print("\nAll experiments completed. Plots are saved in your directory.")