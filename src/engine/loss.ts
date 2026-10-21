/**
 * Loss functions for MiniDeepEngine
 * Supports Regression (MSE) and Classification (Binary & Categorical Cross-Entropy)
 */

import { Matrix, MatrixOps } from './matrix';

export interface LossInterface {
  name: string;
  forward(y_pred: Matrix, y_true: Matrix): number;
  backward(dvalues: Matrix, y_true: Matrix): Matrix;
  toJSON?(): Record<string, unknown>;
}

export class Loss_MSE implements LossInterface {
  name = 'MSE';
  sample_losses: number[] = [];
  dvalues: Matrix = [];

  forward(y_pred: Matrix, y_true: Matrix): number {
    const [rows, cols] = MatrixOps.shape(y_pred);
    this.sample_losses = new Array(rows);
    let total = 0;

    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        const diff = y_pred[i][j] - y_true[i][j];
        rowSum += diff * diff;
      }
      this.sample_losses[i] = rowSum / cols;
      total += this.sample_losses[i];
    }

    return total / rows;
  }

  backward(dvalues: Matrix, y_true: Matrix): Matrix {
    const samples = dvalues.length;
    const cols = dvalues[0]?.length || 1;
    // Derivative of MSE: 2 * (y_pred - y_true) / (samples * cols)
    this.dvalues = MatrixOps.map(dvalues, (v, i, j) => {
      return (2 * (v - y_true[i][j])) / (samples * cols);
    });
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Loss_MSE' };
  }
}

export class Loss_BinaryCrossEntropy implements LossInterface {
  name = 'BinaryCrossEntropy';
  dvalues: Matrix = [];

  forward(y_pred: Matrix, y_true: Matrix): number {
    const [rows, cols] = MatrixOps.shape(y_pred);
    let totalLoss = 0;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        // Clip to avoid log(0)
        const p = Math.max(1e-12, Math.min(1 - 1e-12, y_pred[i][j]));
        const y = y_true[i][j];
        totalLoss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      }
    }

    return totalLoss / (rows * cols);
  }

  backward(dvalues: Matrix, y_true: Matrix): Matrix {
    const samples = dvalues.length;
    const cols = dvalues[0]?.length || 1;

    this.dvalues = MatrixOps.map(dvalues, (v, i, j) => {
      const p = Math.max(1e-12, Math.min(1 - 1e-12, v));
      const y = y_true[i][j];
      return (-((y / p) - ((1 - y) / (1 - p)))) / (samples * cols);
    });

    return this.dvalues;
  }

  toJSON() {
    return { type: 'Loss_BinaryCrossEntropy' };
  }
}

export class Loss_CategoricalCrossEntropy implements LossInterface {
  name = 'CategoricalCrossEntropy';
  dvalues: Matrix = [];

  forward(y_pred: Matrix, y_true: Matrix): number {
    const [rows, cols] = MatrixOps.shape(y_pred);
    let totalLoss = 0;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const p = Math.max(1e-12, Math.min(1.0, y_pred[i][j]));
        totalLoss += -(y_true[i][j] * Math.log(p));
      }
    }

    return totalLoss / rows;
  }

  backward(dvalues: Matrix, y_true: Matrix): Matrix {
    const samples = dvalues.length;
    this.dvalues = MatrixOps.map(dvalues, (v, i, j) => {
      const p = Math.max(1e-12, Math.min(1.0, v));
      return (-y_true[i][j] / p) / samples;
    });
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Loss_CategoricalCrossEntropy' };
  }
}
