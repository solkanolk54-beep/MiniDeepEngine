/**
 * Layers implementation for MiniDeepEngine
 * Dense (Fully Connected) and Dropout layers
 */

import { LayerInterface } from './activations';
import { Matrix, MatrixOps } from './matrix';

export class DenseLayer implements LayerInterface {
  name = 'DenseLayer';
  n_inputs: number;
  n_neurons: number;
  weights: Matrix;
  biases: Matrix;

  // Stored values during forward pass
  inputs: Matrix = [];
  z: Matrix = [];
  output: Matrix = [];

  // Gradients computed during backward pass
  dweights: Matrix = [];
  dbiases: Matrix = [];
  dvalues: Matrix = []; // gradient with respect to inputs

  // Optimizer state tracking (Adam / Momentum)
  weight_momentums?: Matrix;
  weight_cache?: Matrix;
  bias_momentums?: Matrix;
  bias_cache?: Matrix;

  // Regularization options
  l2_weight: number;

  constructor(n_inputs: number, n_neurons: number, initScale?: number, l2_weight = 0) {
    this.n_inputs = n_inputs;
    this.n_neurons = n_neurons;
    this.l2_weight = l2_weight;

    // Xavier/He initialization heuristic or default scale
    const scale = initScale !== undefined ? initScale : Math.sqrt(2.0 / (n_inputs + n_neurons));
    this.weights = MatrixOps.randn(n_inputs, n_neurons, scale);
    this.biases = MatrixOps.zeros(1, n_neurons);
  }

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    this.z = MatrixOps.addBias(MatrixOps.dot(inputs, this.weights), this.biases);
    this.output = this.z;
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    // Chain rule:
    // dW = inputs^T · dvalues
    // db = sum(dvalues, axis=0)
    // dinputs = dvalues · weights^T
    const inputsT = MatrixOps.transpose(this.inputs);
    this.dweights = MatrixOps.dot(inputsT, dvalues);

    // Apply L2 regularization gradient if configured
    if (this.l2_weight > 0) {
      this.dweights = MatrixOps.map(this.dweights, (dw, i, j) => {
        return dw + 2 * this.l2_weight * this.weights[i][j];
      });
    }

    this.dbiases = MatrixOps.sumAxis0(dvalues);

    const weightsT = MatrixOps.transpose(this.weights);
    this.dvalues = MatrixOps.dot(dvalues, weightsT);

    return this.dvalues;
  }

  toJSON(): Record<string, unknown> {
    return {
      type: 'DenseLayer',
      n_inputs: this.n_inputs,
      n_neurons: this.n_neurons,
      weights: this.weights,
      biases: this.biases,
      l2_weight: this.l2_weight,
    };
  }

  fromJSON(data: Record<string, unknown>): void {
    if (data.n_inputs) this.n_inputs = Number(data.n_inputs);
    if (data.n_neurons) this.n_neurons = Number(data.n_neurons);
    if (Array.isArray(data.weights)) this.weights = data.weights as Matrix;
    if (Array.isArray(data.biases)) this.biases = data.biases as Matrix;
    if (data.l2_weight !== undefined) this.l2_weight = Number(data.l2_weight);
  }
}

export class DropoutLayer implements LayerInterface {
  name = 'Dropout';
  rate: number;
  mask: Matrix = [];
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  constructor(rate = 0.2) {
    this.rate = rate;
  }

  forward(inputs: Matrix, isTraining = true): Matrix {
    this.inputs = inputs;
    if (!isTraining || this.rate <= 0) {
      this.output = MatrixOps.clone(inputs);
      return this.output;
    }

    const [rows, cols] = MatrixOps.shape(inputs);
    this.mask = MatrixOps.zeros(rows, cols);
    const keepProb = 1 - this.rate;

    this.output = MatrixOps.map(inputs, (v, i, j) => {
      const keep = Math.random() < keepProb ? 1 : 0;
      this.mask[i][j] = keep / keepProb;
      return v * this.mask[i][j];
    });

    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    if (this.mask.length === 0) {
      this.dvalues = MatrixOps.clone(dvalues);
      return this.dvalues;
    }
    this.dvalues = MatrixOps.multiply(dvalues, this.mask);
    return this.dvalues;
  }

  toJSON(): Record<string, unknown> {
    return {
      type: 'DropoutLayer',
      rate: this.rate,
    };
  }
}
