/**
 * Optimization algorithms for MiniDeepEngine
 * Includes SGD (with optional momentum) and Adam (Adaptive Moment Estimation)
 */

import { DenseLayer } from './layers';
import { MatrixOps } from './matrix';

export interface OptimizerInterface {
  name: string;
  learning_rate: number;
  iterations: number;
  update_params(layer: DenseLayer): void;
  toJSON?(): Record<string, unknown>;
}

export class Optimizer_SGD implements OptimizerInterface {
  name = 'SGD';
  learning_rate: number;
  momentum: number;
  iterations = 0;

  constructor(learning_rate = 0.05, momentum = 0.0) {
    this.learning_rate = learning_rate;
    this.momentum = momentum;
  }

  update_params(layer: DenseLayer): void {
    if (!layer.dweights || !layer.dbiases) return;
    this.iterations++;

    if (this.momentum > 0) {
      if (!layer.weight_momentums) {
        layer.weight_momentums = MatrixOps.zeros(layer.n_inputs, layer.n_neurons);
        layer.bias_momentums = MatrixOps.zeros(1, layer.n_neurons);
      }

      const wm = layer.weight_momentums;
      const bm = layer.bias_momentums!;

      // v = momentum * v - lr * grad
      layer.weight_momentums = MatrixOps.map(wm, (v, i, j) => {
        return this.momentum * v - this.learning_rate * layer.dweights[i][j];
      });
      layer.bias_momentums = MatrixOps.map(bm, (v, i, j) => {
        return this.momentum * v - this.learning_rate * layer.dbiases[i][j];
      });

      // w = w + v
      layer.weights = MatrixOps.map(layer.weights, (w, i, j) => {
        return w + layer.weight_momentums![i][j];
      });
      layer.biases = MatrixOps.map(layer.biases, (b, i, j) => {
        return b + layer.bias_momentums![i][j];
      });
    } else {
      // Standard SGD: w = w - lr * dw
      layer.weights = MatrixOps.map(layer.weights, (w, i, j) => {
        return w - this.learning_rate * layer.dweights[i][j];
      });
      layer.biases = MatrixOps.map(layer.biases, (b, i, j) => {
        return b - this.learning_rate * layer.dbiases[i][j];
      });
    }
  }

  toJSON() {
    return {
      type: 'Optimizer_SGD',
      learning_rate: this.learning_rate,
      momentum: this.momentum,
    };
  }
}

export class Optimizer_Adam implements OptimizerInterface {
  name = 'Adam';
  learning_rate: number;
  beta_1: number;
  beta_2: number;
  epsilon: number;
  iterations = 0;

  constructor(learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8) {
    this.learning_rate = learning_rate;
    this.beta_1 = beta_1;
    this.beta_2 = beta_2;
    this.epsilon = epsilon;
  }

  update_params(layer: DenseLayer): void {
    if (!layer.dweights || !layer.dbiases) return;

    if (!layer.weight_momentums || !layer.weight_cache || !layer.bias_momentums || !layer.bias_cache) {
      layer.weight_momentums = MatrixOps.zeros(layer.n_inputs, layer.n_neurons);
      layer.weight_cache = MatrixOps.zeros(layer.n_inputs, layer.n_neurons);
      layer.bias_momentums = MatrixOps.zeros(1, layer.n_neurons);
      layer.bias_cache = MatrixOps.zeros(1, layer.n_neurons);
    }

    const wm = layer.weight_momentums;
    const wc = layer.weight_cache;
    const bm = layer.bias_momentums;
    const bc = layer.bias_cache;

    this.iterations++;
    const t = this.iterations;

    // --- 1. Weights update ---
    const beta1Correction = 1 - Math.pow(this.beta_1, t);
    const beta2Correction = 1 - Math.pow(this.beta_2, t);

    layer.weight_momentums = MatrixOps.map(wm, (m, i, j) => {
      const dw = layer.dweights[i][j];
      return this.beta_1 * m + (1 - this.beta_1) * dw;
    });

    layer.weight_cache = MatrixOps.map(wc, (v, i, j) => {
      const dw = layer.dweights[i][j];
      return this.beta_2 * v + (1 - this.beta_2) * (dw * dw);
    });

    // weights update with bias correction
    layer.weights = MatrixOps.map(layer.weights, (w, i, j) => {
      const mCorrected = layer.weight_momentums![i][j] / beta1Correction;
      const vCorrected = layer.weight_cache![i][j] / beta2Correction;
      return w - (this.learning_rate * mCorrected) / (Math.sqrt(vCorrected) + this.epsilon);
    });

    // --- 2. Biases update ---
    layer.bias_momentums = MatrixOps.map(bm, (m, i, j) => {
      const db = layer.dbiases[i][j];
      return this.beta_1 * m + (1 - this.beta_1) * db;
    });

    layer.bias_cache = MatrixOps.map(bc, (v, i, j) => {
      const db = layer.dbiases[i][j];
      return this.beta_2 * v + (1 - this.beta_2) * (db * db);
    });

    layer.biases = MatrixOps.map(layer.biases, (b, i, j) => {
      const mCorrected = layer.bias_momentums![i][j] / beta1Correction;
      const vCorrected = layer.bias_cache![i][j] / beta2Correction;
      return b - (this.learning_rate * mCorrected) / (Math.sqrt(vCorrected) + this.epsilon);
    });
  }

  toJSON() {
    return {
      type: 'Optimizer_Adam',
      learning_rate: this.learning_rate,
      beta_1: this.beta_1,
      beta_2: this.beta_2,
      epsilon: this.epsilon,
    };
  }
}
