/**
 * MiniDeepEngine NeuralNetwork Model Architecture
 * Implements Keras-like model API: add, set, forward, backward, fit, save, load
 */

import { LayerInterface, Activation_ReLU, Activation_Sigmoid, Activation_Tanh, Activation_LeakyReLU, Activation_Linear, Activation_Softmax } from './activations';
import { DenseLayer, DropoutLayer } from './layers';
import { LossInterface, Loss_MSE, Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy } from './loss';
import { Matrix, MatrixOps } from './matrix';
import { OptimizerInterface, Optimizer_Adam, Optimizer_SGD } from './optimizers';

export interface EvaluationMetrics {
  loss: number;
  accuracy?: number;
  f1Score?: number;
  rmse?: number;
}

export class NeuralNetwork {
  layers: LayerInterface[] = [];
  loss_function: LossInterface | null = null;
  optimizer: OptimizerInterface | null = null;
  loss_history: number[] = [];
  metric_history: number[] = [];

  add(layer: LayerInterface): void {
    this.layers.push(layer);
  }

  set(loss: LossInterface, optimizer: OptimizerInterface): void {
    this.loss_function = loss;
    this.optimizer = optimizer;
  }

  forward(X: Matrix, isTraining = false): Matrix {
    let output = X;
    for (const layer of this.layers) {
      output = layer.forward(output, isTraining);
    }
    return output;
  }

  backward(prediction: Matrix, y_true: Matrix): void {
    if (!this.loss_function) throw new Error('Loss function not configured');
    
    // 1. Loss gradients
    let dvalues = this.loss_function.backward(prediction, y_true);

    // 2. Backward pass in reverse order through layers
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];
      dvalues = layer.backward(dvalues);
    }
  }

  optimize(): void {
    if (!this.optimizer) throw new Error('Optimizer not configured');
    for (const layer of this.layers) {
      if (layer instanceof DenseLayer) {
        this.optimizer.update_params(layer);
      }
    }
  }

  /**
   * Run a single step of forward pass, loss calculation, backward propagation, and optimization.
   */
  step(X: Matrix, y: Matrix): { loss: number; prediction: Matrix; metrics: EvaluationMetrics } {
    // 1. Forward
    const prediction = this.forward(X, true);

    // 2. Loss
    if (!this.loss_function) throw new Error('Loss function not set');
    const loss = this.loss_function.forward(prediction, y);
    this.loss_history.push(loss);

    // 3. Backward
    this.backward(prediction, y);

    // 4. Optimize
    this.optimize();

    // 5. Compute metrics
    const metrics = this.evaluate(X, y);
    if (metrics.accuracy !== undefined) {
      this.metric_history.push(metrics.accuracy);
    } else if (metrics.rmse !== undefined) {
      this.metric_history.push(metrics.rmse);
    }

    return { loss, prediction, metrics };
  }

  evaluate(X: Matrix, y: Matrix): EvaluationMetrics {
    const preds = this.forward(X, false);
    const loss = this.loss_function ? this.loss_function.forward(preds, y) : 0;
    const [rows, cols] = MatrixOps.shape(preds);

    // Determine if classification or regression
    const isClassification = this.loss_function instanceof Loss_BinaryCrossEntropy ||
      this.loss_function instanceof Loss_CategoricalCrossEntropy ||
      (cols === 1 && y.every(row => row[0] === 0 || row[0] === 1));

    if (isClassification) {
      let correct = 0;
      let tp = 0;
      let fp = 0;
      let fn = 0;

      if (cols === 1) {
        // Binary classification
        for (let i = 0; i < rows; i++) {
          const predictedClass = preds[i][0] >= 0.5 ? 1 : 0;
          const actualClass = y[i][0] >= 0.5 ? 1 : 0;
          if (predictedClass === actualClass) correct++;
          if (predictedClass === 1 && actualClass === 1) tp++;
          if (predictedClass === 1 && actualClass === 0) fp++;
          if (predictedClass === 0 && actualClass === 1) fn++;
        }
      } else {
        // Multi-class classification (argmax)
        for (let i = 0; i < rows; i++) {
          let maxPredIdx = 0;
          let maxPredVal = -Infinity;
          let maxActualIdx = 0;
          let maxActualVal = -Infinity;
          for (let j = 0; j < cols; j++) {
            if (preds[i][j] > maxPredVal) {
              maxPredVal = preds[i][j];
              maxPredIdx = j;
            }
            if (y[i][j] > maxActualVal) {
              maxActualVal = y[i][j];
              maxActualIdx = j;
            }
          }
          if (maxPredIdx === maxActualIdx) correct++;
        }
      }

      const accuracy = correct / rows;
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

      return { loss, accuracy, f1Score };
    } else {
      // Numerical regression (calculate RMSE)
      let sumSq = 0;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const diff = preds[i][j] - y[i][j];
          sumSq += diff * diff;
        }
      }
      const rmse = Math.sqrt(sumSq / (rows * cols));
      return { loss, rmse };
    }
  }

  fit(
    X: Matrix,
    y: Matrix,
    epochs = 1000,
    print_every = 200,
    onEpoch?: (epoch: number, loss: number, metrics: EvaluationMetrics) => void
  ): void {
    this.loss_history = [];
    this.metric_history = [];

    for (let epoch = 0; epoch <= epochs; epoch++) {
      const { loss, metrics } = this.step(X, y);

      if (onEpoch && (epoch % print_every === 0 || epoch === epochs)) {
        onEpoch(epoch, loss, metrics);
      }
    }
  }

  /**
   * Save model to JSON string
   */
  toJSON(): string {
    return JSON.stringify({
      version: '1.0.0',
      loss_function: this.loss_function?.toJSON?.() || null,
      optimizer: this.optimizer?.toJSON?.() || null,
      layers: this.layers.map(layer => layer.toJSON?.() || { type: layer.name }),
    }, null, 2);
  }

  /**
   * Load model from JSON representation
   */
  static fromJSON(jsonStr: string): NeuralNetwork {
    const data = JSON.parse(jsonStr);
    const net = new NeuralNetwork();

    // Reconstruct layers
    for (const layerData of data.layers) {
      if (layerData.type === 'DenseLayer') {
        const dense = new DenseLayer(layerData.n_inputs, layerData.n_neurons, undefined, layerData.l2_weight);
        dense.fromJSON(layerData);
        net.add(dense);
      } else if (layerData.type === 'Activation_ReLU') {
        net.add(new Activation_ReLU());
      } else if (layerData.type === 'Activation_Sigmoid') {
        net.add(new Activation_Sigmoid());
      } else if (layerData.type === 'Activation_Tanh') {
        net.add(new Activation_Tanh());
      } else if (layerData.type === 'Activation_LeakyReLU') {
        net.add(new Activation_LeakyReLU(layerData.alpha));
      } else if (layerData.type === 'Activation_Linear') {
        net.add(new Activation_Linear());
      } else if (layerData.type === 'Activation_Softmax') {
        net.add(new Activation_Softmax());
      } else if (layerData.type === 'DropoutLayer') {
        net.add(new DropoutLayer(layerData.rate));
      }
    }

    // Reconstruct loss
    let loss: LossInterface = new Loss_MSE();
    if (data.loss_function?.type === 'Loss_BinaryCrossEntropy') {
      loss = new Loss_BinaryCrossEntropy();
    } else if (data.loss_function?.type === 'Loss_CategoricalCrossEntropy') {
      loss = new Loss_CategoricalCrossEntropy();
    }

    // Reconstruct optimizer
    let optimizer: OptimizerInterface = new Optimizer_Adam();
    if (data.optimizer?.type === 'Optimizer_SGD') {
      optimizer = new Optimizer_SGD(data.optimizer.learning_rate, data.optimizer.momentum);
    } else if (data.optimizer?.type === 'Optimizer_Adam') {
      optimizer = new Optimizer_Adam(
        data.optimizer.learning_rate,
        data.optimizer.beta_1,
        data.optimizer.beta_2,
        data.optimizer.epsilon
      );
    }

    net.set(loss, optimizer);
    return net;
  }

  /**
   * Generates runnable Python code matching the user's framework
   */
  exportPythonCode(datasetX: Matrix, datasetY: Matrix, epochs = 1000): string {
    const layersCode = this.layers.map(layer => {
      if (layer instanceof DenseLayer) {
        return `    model.add(DenseLayer(n_inputs=${layer.n_inputs}, n_neurons=${layer.n_neurons}))`;
      }
      if (layer instanceof Activation_ReLU) return `    model.add(Activation_ReLU())`;
      if (layer instanceof Activation_Sigmoid) return `    model.add(Activation_Sigmoid())`;
      if (layer instanceof Activation_Tanh) return `    model.add(Activation_Tanh())`;
      if (layer instanceof Activation_LeakyReLU) return `    model.add(Activation_LeakyReLU(alpha=${layer.alpha}))`;
      if (layer instanceof Activation_Linear) return `    model.add(Activation_Linear())`;
      return `    # Unknown layer`;
    }).join('\n');

    let lossCode = 'Loss_MSE()';
    if (this.loss_function instanceof Loss_BinaryCrossEntropy) {
      lossCode = 'Loss_BinaryCrossEntropy()';
    }

    let optCode = 'Optimizer_Adam(learning_rate=0.01)';
    if (this.optimizer instanceof Optimizer_Adam) {
      optCode = `Optimizer_Adam(learning_rate=${this.optimizer.learning_rate}, beta_1=${this.optimizer.beta_1}, beta_2=${this.optimizer.beta_2})`;
    } else if (this.optimizer instanceof Optimizer_SGD) {
      optCode = `Optimizer_SGD(learning_rate=${this.optimizer.learning_rate})`;
    }

    const xStr = JSON.stringify(datasetX);
    const yStr = JSON.stringify(datasetY);

    return `# ==========================================================
# MiniDeepEngine - Python Implementation
# Built from scratch with NumPy (Forward, Activations, Backpropagation, Adam)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- 1. Layers & Activations ---
class DenseLayer:
    """Fully Connected Dense Layer"""
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dvalues = np.dot(dvalues, self.weights.T)
        return self.dvalues

class Activation_ReLU:
    """ReLU Activation: max(0, x)"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0
        return self.dvalues

class Activation_Sigmoid:
    """Sigmoid Activation: 1 / (1 + exp(-x))"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, dvalues):
        self.dvalues = dvalues * self.output * (1 - self.output)
        return self.dvalues

class Activation_Tanh:
    """Tanh Activation"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, dvalues):
        self.dvalues = dvalues * (1 - self.output ** 2)
        return self.dvalues

class Activation_LeakyReLU:
    """LeakyReLU Activation"""
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] *= self.alpha
        return self.dvalues

class Activation_Linear:
    """Linear (Identity) Activation"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
        return self.output

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        return self.dvalues

# --- 2. Loss Functions ---
class Loss_MSE:
    """Mean Squared Error (MSE)"""
    def forward(self, y_pred, y_true):
        self.sample_losses = np.mean((y_pred - y_true) ** 2, axis=-1)
        return np.mean(self.sample_losses)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dvalues = 2 * (dvalues - y_true) / samples
        return self.dvalues

class Loss_BinaryCrossEntropy:
    """Binary Cross Entropy"""
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dvalues = -(y_true / clipped - (1 - y_true) / (1 - clipped)) / samples
        return self.dvalues

# --- 3. Optimizers ---
class Optimizer_SGD:
    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Optimizer_Adam:
    """Adam (Adaptive Moment Estimation) with Bias Correction"""
    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        self.iterations += 1
        t = self.iterations

        # First moment (m_t) & second moment (v_t)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        
        # Bias correction
        weight_m_corr = layer.weight_momentums / (1 - self.beta_1 ** t)
        weight_v_corr = layer.weight_cache / (1 - self.beta_2 ** t)
        
        # Update weights
        layer.weights -= self.learning_rate * weight_m_corr / (np.sqrt(weight_v_corr) + self.epsilon)

        # Biases update
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)
        
        bias_m_corr = layer.bias_momentums / (1 - self.beta_1 ** t)
        bias_v_corr = layer.bias_cache / (1 - self.beta_2 ** t)
        
        layer.biases -= self.learning_rate * bias_m_corr / (np.sqrt(bias_v_corr) + self.epsilon)

# --- 4. Neural Network Engine ---
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.loss_history = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss, optimizer):
        self.loss_function = loss
        self.optimizer = optimizer

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, y_true):
        dvalues = self.loss_function.backward(prediction, y_true)
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def optimize(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                self.optimizer.update_params(layer)

    def fit(self, X, y, epochs=1000, print_every=200):
        print("=" * 45)
        print("   Starting MiniDeepEngine Training   ")
        print("=" * 45)
        self.loss_history = []
        for epoch in range(epochs + 1):
            prediction = self.forward(X)
            loss = self.loss_function.forward(prediction, y)
            self.loss_history.append(loss)
            self.backward(prediction, y)
            self.optimize()
            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        print("=" * 45)
        print("   Training Completed Successfully!   ")
        print("=" * 45)

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, color='blue', linewidth=2, label='Training Loss')
        plt.title('MiniDeepEngine - Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    def save(self, filepath="model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved successfully to: {filepath}")

    @staticmethod
    def load(filepath="model.pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)

# --- 5. Execution Pipeline ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Dataset
    X = np.array(${xStr})
    y = np.array(${yStr})

    # Build model
    model = NeuralNetwork()
${layersCode}

    # Compile with Loss & Optimizer
    model.set(loss=${lossCode}, optimizer=${optCode})

    # Train
    model.fit(X, y, epochs=${epochs}, print_every=${Math.max(1, Math.floor(epochs / 5))})

    # Predict & Compare
    predictions = model.forward(X)
    print("\\nFinal Predictions:\\n", np.round(predictions, 4))
    print("Ground Truth Target:\\n", y)

    # Plot
    model.plot_loss()
`;
  }
}
