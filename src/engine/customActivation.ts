/**
 * Custom and Experimental Activation Functions for Educational & R&D Lab
 * Allows testing novel activation functions (Swish, Mish, GELU, Siren, Custom Formula)
 * and observing their impact on gradient flow and the Loss Curve.
 */

import { LayerInterface } from './activations';
import { Matrix, MatrixOps } from './matrix';

export type CustomActivationType = 'swish' | 'mish' | 'gelu' | 'siren' | 'softplus' | 'custom_expr';

export interface CustomActivationPreset {
  id: CustomActivationType;
  name: string;
  nameAr: string;
  formula: string;
  derivativeFormula: string;
  description: string;
  descriptionAr: string;
}

export const CUSTOM_ACTIVATION_PRESETS: CustomActivationPreset[] = [
  {
    id: 'swish',
    name: 'Swish (SiLU)',
    nameAr: 'سويش (Swish / SiLU)',
    formula: 'f(x) = x · σ(x)',
    derivativeFormula: "f'(x) = σ(x) + x · σ(x) · (1 - σ(x))",
    description: 'Smooth, non-monotonic activation developed by Google Brain that outperforms ReLU in deep architectures.',
    descriptionAr: 'دالة تفعيل ملساء وغير رتيبة ابتكرها باحثو Google Brain لتفوقها على ReLU في الشبكات العميقة.',
  },
  {
    id: 'mish',
    name: 'Mish',
    nameAr: 'ميش (Mish)',
    formula: 'f(x) = x · tanh(ln(1 + eˣ))',
    derivativeFormula: "f'(x) = tanh(softplus(x)) + x · σ(x) · (1 - tanh²(softplus(x)))",
    description: 'Self-regularized smooth activation function allowing slight negative values to preserve information flow.',
    descriptionAr: 'دالة ذاتية التنظيم تسمح بمرور تدرجات سالبة طفيفة لتفادي انقطاع التعلم وتسهيل هبوط المنحدرات.',
  },
  {
    id: 'gelu',
    name: 'GELU (Gaussian Error Linear Unit)',
    nameAr: 'جيلو (GELU - المستخدمة في المحولات Transformer)',
    formula: 'f(x) ≈ 0.5x · (1 + tanh(√(2/π)(x + 0.044715x³)))',
    derivativeFormula: 'Analytical approximation of Gaussian CDF gradient',
    description: 'Standard activation used in modern Large Language Models and Vision Transformers.',
    descriptionAr: 'المعيار المعتمد في النماذج اللغوية الكبيرة ونماذج المحولات (Transformers) لربط التفعيل باحتمالية غاوس.',
  },
  {
    id: 'siren',
    name: 'Siren (Sinusoidal)',
    nameAr: 'دالة الجيب الترددية (Siren / Sine)',
    formula: 'f(x) = sin(x)',
    derivativeFormula: "f'(x) = cos(x)",
    description: 'Periodic activation function popular in implicit neural representations and wave fitting.',
    descriptionAr: 'دالة تفعيل دورية ممتازة لتمثيل الإشارات والترددات والخرائط الضمنية والتقريب الدقيق.',
  },
  {
    id: 'softplus',
    name: 'Softplus',
    nameAr: 'سوفت بلس (Softplus)',
    formula: 'f(x) = ln(1 + eˣ)',
    derivativeFormula: "f'(x) = 1 / (1 + e⁻ˣ) = σ(x)",
    description: 'Smooth approximation to the rectifier ReLU function with derivative equal to Sigmoid.',
    descriptionAr: 'تقريب ناعم وأملس لدالة ReLU، وتتميز بأن مشتقتها التحليلية تساوي تماماً دالة Sigmoid.',
  },
];

export class Activation_Custom implements LayerInterface {
  name: string;
  presetType: CustomActivationType;
  customExpr: string;
  inputs: Matrix = [];
  output: Matrix = [];
  dvalues: Matrix = [];

  constructor(presetType: CustomActivationType = 'swish', customExpr = 'x / (1 + Math.abs(x))') {
    this.presetType = presetType;
    this.customExpr = customExpr;
    this.name = `Custom_${presetType.toUpperCase()}`;
  }

  forward(inputs: Matrix): Matrix {
    this.inputs = inputs;

    switch (this.presetType) {
      case 'swish': {
        this.output = MatrixOps.map(inputs, x => {
          const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
          return x * sig;
        });
        break;
      }
      case 'mish': {
        this.output = MatrixOps.map(inputs, x => {
          const sp = Math.log(1 + Math.exp(Math.max(-500, Math.min(50, x))));
          return x * Math.tanh(sp);
        });
        break;
      }
      case 'gelu': {
        this.output = MatrixOps.map(inputs, x => {
          const c = Math.sqrt(2 / Math.PI);
          return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * Math.pow(x, 3))));
        });
        break;
      }
      case 'siren': {
        this.output = MatrixOps.map(inputs, x => Math.sin(x));
        break;
      }
      case 'softplus': {
        this.output = MatrixOps.map(inputs, x => {
          return Math.log(1 + Math.exp(Math.max(-500, Math.min(50, x))));
        });
        break;
      }
      case 'custom_expr':
      default: {
        // Safe evaluation of custom math expression with fallback
        this.output = MatrixOps.map(inputs, x => {
          try {
            // Function evaluation with math context
            const fn = new Function('x', 'Math', `try { return (${this.customExpr}); } catch(e) { return x; }`);
            const val = fn(x, Math);
            return typeof val === 'number' && !isNaN(val) ? val : x;
          } catch {
            return x;
          }
        });
        break;
      }
    }

    return this.output;
  }

  backward(dvalues: Matrix): Matrix {
    const h = 1e-5;

    switch (this.presetType) {
      case 'swish': {
        this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
          const x = this.inputs[i][j];
          const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
          const grad = sig + x * sig * (1 - sig);
          return dv * grad;
        });
        break;
      }
      case 'mish': {
        this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
          const x = this.inputs[i][j];
          const sp = Math.log(1 + Math.exp(Math.max(-500, Math.min(50, x))));
          const tanhSp = Math.tanh(sp);
          const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
          const grad = tanhSp + x * sig * (1 - tanhSp * tanhSp);
          return dv * grad;
        });
        break;
      }
      case 'gelu': {
        this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
          const x = this.inputs[i][j];
          // Numerical derivative for stability
          const c = Math.sqrt(2 / Math.PI);
          const f_plus = 0.5 * (x + h) * (1 + Math.tanh(c * ((x + h) + 0.044715 * Math.pow(x + h, 3))));
          const f_minus = 0.5 * (x - h) * (1 + Math.tanh(c * ((x - h) + 0.044715 * Math.pow(x - h, 3))));
          const grad = (f_plus - f_minus) / (2 * h);
          return dv * grad;
        });
        break;
      }
      case 'siren': {
        this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
          const x = this.inputs[i][j];
          return dv * Math.cos(x);
        });
        break;
      }
      case 'softplus': {
        this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
          const x = this.inputs[i][j];
          const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
          return dv * sig;
        });
        break;
      }
      case 'custom_expr':
      default: {
        // High-precision symmetric numerical derivative (f(x+h) - f(x-h)) / 2h
        this.dvalues = MatrixOps.map(dvalues, (dv, i, j) => {
          const x = this.inputs[i][j];
          try {
            const fn = new Function('x', 'Math', `try { return (${this.customExpr}); } catch(e) { return x; }`);
            const y1 = fn(x + h, Math);
            const y2 = fn(x - h, Math);
            const numGrad = (y1 - y2) / (2 * h);
            return dv * (typeof numGrad === 'number' && !isNaN(numGrad) ? numGrad : 1);
          } catch {
            return dv;
          }
        });
        break;
      }
    }

    return this.dvalues;
  }

  evalScalar(x: number): number {
    const res = this.forward([[x]]);
    return res[0][0];
  }

  evalDerivativeScalar(x: number): number {
    this.forward([[x]]);
    const grad = this.backward([[1.0]]);
    return grad[0][0];
  }

  toJSON() {
    return {
      type: 'Activation_Custom',
      presetType: this.presetType,
      customExpr: this.customExpr,
    };
  }
}
