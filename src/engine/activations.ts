/**
 * Activation functions and their mathematical derivatives (backward pass)
 * for MiniDeepEngine.
 */

import { Matrix, MatrixOps } from './matrix';

export interface LayerInterface {
  name: string;
  forward(inputs: Matrix, isTraining?: boolean): Matrix;
  backward(dvalues: Matrix): Matrix;
  toJSON?(): Record<string, unknown>;
  fromJSON?(data: Record<string, unknown>): void;
}

export class Activation_ReLU implements LayerInterface {
  name = 'ReLU';
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    this.output = MatrixOps.map(inputs, v => Math.max(0, v));
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    // ReLU derivative: 1 if input > 0 else 0
    this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
      return this.inputs[i][j] > 0 ? dv : 0;
    });
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Activation_ReLU' };
  }
}

export class Activation_Sigmoid implements LayerInterface {
  name = 'Sigmoid';
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    // clip inputs between -500 and 500 to prevent numerical overflow
    this.output = MatrixOps.map(inputs, v => {
      const clipped = Math.max(-500, Math.min(500, v));
      return 1 / (1 + Math.exp(-clipped));
    });
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    // Sigmoid derivative: output * (1 - output)
    this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
      const o = this.output[i][j];
      return dv * o * (1 - o);
    });
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Activation_Sigmoid' };
  }
}

export class Activation_Tanh implements LayerInterface {
  name = 'Tanh';
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    this.output = MatrixOps.map(inputs, v => Math.tanh(v));
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    // Tanh derivative: 1 - output^2
    this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
      const o = this.output[i][j];
      return dv * (1 - o * o);
    });
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Activation_Tanh' };
  }
}

export class Activation_LeakyReLU implements LayerInterface {
  name = 'LeakyReLU';
  alpha: number;
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  constructor(alpha = 0.01) {
    this.alpha = alpha;
  }

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    this.output = MatrixOps.map(inputs, v => (v > 0 ? v : this.alpha * v));
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    // LeakyReLU derivative: 1 if input > 0 else alpha
    this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
      return this.inputs[i][j] > 0 ? dv : dv * this.alpha;
    });
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Activation_LeakyReLU', alpha: this.alpha };
  }
}

export class Activation_Linear implements LayerInterface {
  name = 'Linear';
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    this.output = MatrixOps.clone(inputs);
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    this.dvalues = MatrixOps.clone(dvalues);
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Activation_Linear' };
  }
}

export class Activation_Softmax implements LayerInterface {
  name = 'Softmax';
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;
    const [rows, cols] = MatrixOps.shape(inputs);
    this.output = new Array(rows);

    for (let i = 0; i < rows; i++) {
      // Subtract max for numerical stability
      let maxVal = -Infinity;
      for (let j = 0; j < cols; j++) {
        if (inputs[i][j] > maxVal) maxVal = inputs[i][j];
      }

      let expSum = 0;
      const exps = new Array(cols);
      for (let j = 0; j < cols; j++) {
        exps[j] = Math.exp(inputs[i][j] - maxVal);
        expSum += exps[j];
      }

      this.output[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        this.output[i][j] = exps[j] / (expSum || 1e-15);
      }
    }
    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    const [rows, cols] = MatrixOps.shape(dvalues);
    this.dvalues = MatrixOps.zeros(rows, cols);

    for (let i = 0; i < rows; i++) {
      const s = this.output[i];
      const d = dvalues[i];
      for (let j = 0; j < cols; j++) {
        let grad = 0;
        for (let k = 0; k < cols; k++) {
          const delta = j === k ? 1 : 0;
          grad += d[k] * s[j] * (delta - s[k]);
        }
        this.dvalues[i][j] = grad;
      }
    }
    return this.dvalues;
  }

  toJSON() {
    return { type: 'Activation_Softmax' };
  }
}
