/**
 * Matrix & Tensor mathematical utilities for MiniDeepEngine
 * Pure TypeScript implementation matching NumPy numerical semantics
 */

export type Matrix = number[][];

export class MatrixOps {
  /** Create matrix of shape [rows, cols] filled with val */
  static fill(rows: number, cols: number, val: number): Matrix {
    const res: Matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
      res[i] = new Array(cols).fill(val);
    }
    return res;
  }

  /** Create matrix of zeros */
  static zeros(rows: number, cols: number): Matrix {
    return this.fill(rows, cols, 0);
  }

  /** Create matrix with standard normal distribution (Box-Muller transform) */
  static randn(rows: number, cols: number, scale = 0.01): Matrix {
    const res: Matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
      res[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.max(1e-15, Math.random());
        const u2 = Math.random();
        const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
        res[i][j] = randStdNormal * scale;
      }
    }
    return res;
  }

  /** Deep clone matrix */
  static clone(a: Matrix): Matrix {
    return a.map(row => [...row]);
  }

  /** Matrix dimensions [rows, cols] */
  static shape(a: Matrix): [number, number] {
    return [a.length, a[0]?.length || 0];
  }

  /** Matrix transpose A^T */
  static transpose(a: Matrix): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = new Array(cols);
    for (let j = 0; j < cols; j++) {
      res[j] = new Array(rows);
      for (let i = 0; i < rows; i++) {
        res[j][i] = a[i][j];
      }
    }
    return res;
  }

  /** Matrix dot product: (M x K) dot (K x N) -> (M x N) */
  static dot(a: Matrix, b: Matrix): Matrix {
    const [rowsA, colsA] = this.shape(a);
    const [rowsB, colsB] = this.shape(b);
    if (colsA !== rowsB) {
      throw new Error(`Matrix dot dimension mismatch: [${rowsA}, ${colsA}] dot [${rowsB}, ${colsB}]`);
    }

    const res: Matrix = this.zeros(rowsA, colsB);
    for (let i = 0; i < rowsA; i++) {
      const aRow = a[i];
      const resRow = res[i];
      for (let k = 0; k < colsA; k++) {
        const aVal = aRow[k];
        const bRow = b[k];
        for (let j = 0; j < colsB; j++) {
          resRow[j] += aVal * bRow[j];
        }
      }
    }
    return res;
  }

  /** Broadcast addition of biases: a (M x N) + b (1 x N) -> (M x N) */
  static addBias(a: Matrix, bias: Matrix): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = new Array(rows);
    const bRow = bias[0];
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      const aRow = a[i];
      for (let j = 0; j < cols; j++) {
        row[j] = aRow[j] + bRow[j];
      }
      res[i] = row;
    }
    return res;
  }

  /** Element-wise subtraction: a - b */
  static subtract(a: Matrix, b: Matrix): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
      res[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        res[i][j] = a[i][j] - b[i][j];
      }
    }
    return res;
  }

  /** Element-wise multiplication: a * b */
  static multiply(a: Matrix, b: Matrix): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
      res[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        res[i][j] = a[i][j] * b[i][j];
      }
    }
    return res;
  }

  /** Element-wise scalar multiplication: a * scalar */
  static scale(a: Matrix, scalar: number): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
      res[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        res[i][j] = a[i][j] * scalar;
      }
    }
    return res;
  }

  /** Sum over axis 0 (columns sum, keepdims=true -> 1 x cols) */
  static sumAxis0(a: Matrix): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = [new Array(cols).fill(0)];
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        res[0][j] += a[i][j];
      }
    }
    return res;
  }

  /** Clip matrix values to [min, max] */
  static clip(a: Matrix, min: number, max: number): Matrix {
    return a.map(row => row.map(v => Math.max(min, Math.min(max, v))));
  }

  /** Mean of all elements in matrix */
  static mean(a: Matrix): number {
    const [rows, cols] = this.shape(a);
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        sum += a[i][j];
      }
    }
    return sum / (rows * cols);
  }

  /** Apply unary function element-wise */
  static map(a: Matrix, fn: (v: number, i: number, j: number) => number): Matrix {
    const [rows, cols] = this.shape(a);
    const res: Matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
      res[i] = new Array(cols);
      for (let j = 0; j < cols; j++) {
        res[i][j] = fn(a[i][j], i, j);
      }
    }
    return res;
  }
}
