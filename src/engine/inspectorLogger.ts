/**
 * Math Inspector Logger for MiniDeepEngine
 * Logs and formats detailed internal state under the hood:
 * - Forward pass matrices (inputs, weights, biases, outputs)
 * - Loss and upstream error signal
 * - Backpropagation Chain Rule gradients (dweights, dbiases, dinputs)
 * - Optimizer updates (Adam moments or SGD steps)
 */

import { Matrix } from './matrix';

export interface InspectorLayerLog {
  layerName: string;
  layerType: string;
  inputs: Matrix;
  weights?: Matrix;
  biases?: Matrix;
  output: Matrix;
  dweights?: Matrix;
  dbiases?: Matrix;
  dinputs?: Matrix;
  updatedWeights?: Matrix;
  updatedBiases?: Matrix;
  weightGradientsNorm?: number;
}

export interface InspectorEpochLog {
  epoch: number;
  timestamp: number;
  loss: number;
  predictions: Matrix;
  targets: Matrix;
  lossGradient: Matrix;
  layers: InspectorLayerLog[];
  formattedSummary: string;
}

export class MathInspectorLogger {
  logs: InspectorEpochLog[] = [];
  maxLogs: number;
  enabled: boolean = true;

  constructor(maxLogs = 50) {
    this.maxLogs = maxLogs;
  }

  clear(): void {
    this.logs = [];
  }

  logEpoch(
    epoch: number,
    loss: number,
    predictions: Matrix,
    targets: Matrix,
    lossGradient: Matrix,
    layers: InspectorLayerLog[]
  ): InspectorEpochLog {
    const formattedSummary = this.formatTerminalText(epoch, loss, predictions, targets, lossGradient, layers);
    
    const entry: InspectorEpochLog = {
      epoch,
      timestamp: Date.now(),
      loss,
      predictions,
      targets,
      lossGradient,
      layers,
      formattedSummary,
    };

    if (this.logs.length >= this.maxLogs) {
      this.logs.shift();
    }
    this.logs.push(entry);

    return entry;
  }

  /**
   * Generates exact Python terminal style inspection output as requested by user:
   * --- Epoch 1 ---
   * [1] Forward Pass -> Inputs: [...], Prediction: [...], Loss (MSE): ...
   * [2] Backpropagation -> Gradient of Loss w.r.t Weights (dWeights): ...
   *     Gradient w.r.t Biases (dBiases): ...
   * [3] Weights Updated -> New Weights: ...
   */
  private formatTerminalText(
    epoch: number,
    loss: number,
    predictions: Matrix,
    targets: Matrix,
    lossGradient: Matrix,
    layers: InspectorLayerLog[]
  ): string {
    const lines: string[] = [];
    lines.push(`--- Epoch ${epoch} ---`);

    // [1] Forward Pass
    const samplePred = predictions.slice(0, 3).map(r => r.map(v => Number(v.toFixed(4))));
    const sampleTargets = targets.slice(0, 3).map(r => r.map(v => Number(v.toFixed(4))));
    lines.push(`[1] Forward Pass -> Samples: ${targets.length}, Predictions: ${JSON.stringify(samplePred)}, Loss: ${loss.toFixed(6)}`);

    // [2] Backpropagation & Chain Rule
    const sampleLossGrad = lossGradient.slice(0, 3).map(r => r.map(v => Number(v.toFixed(4))));
    lines.push(`[2] Loss Gradient (dValues from Loss Function): ${JSON.stringify(sampleLossGrad)}`);

    layers.forEach((l, idx) => {
      lines.push(`    Layer [${idx + 1}] "${l.layerName}" (${l.layerType}):`);
      if (l.dweights) {
        const dwStr = JSON.stringify(l.dweights.slice(0, 2).map(r => r.slice(0, 4).map(v => Number(v.toFixed(4)))));
        lines.push(`      ├─ Gradient w.r.t Weights (dWeights = Xᵀ • dZ):\n         ${dwStr}`);
      }
      if (l.dbiases) {
        const dbStr = JSON.stringify(l.dbiases[0]?.slice(0, 4).map(v => Number(v.toFixed(4))) || []);
        lines.push(`      ├─ Gradient w.r.t Biases (dBiases = Σ dZ): ${dbStr}`);
      }
      if (l.dinputs) {
        const dxStr = JSON.stringify(l.dinputs.slice(0, 2).map(r => r.slice(0, 4).map(v => Number(v.toFixed(4)))));
        lines.push(`      └─ Backpropagated Gradient to Inputs (dX = dZ • Wᵀ): ${dxStr}`);
      }
    });

    // [3] Optimizer Update
    const denseLayer = layers.find(l => l.updatedWeights);
    if (denseLayer && denseLayer.updatedWeights) {
      const newWStr = JSON.stringify(denseLayer.updatedWeights.slice(0, 2).map(r => r.slice(0, 4).map(v => Number(v.toFixed(4)))));
      lines.push(`[3] Parameter Update -> Updated Weights for "${denseLayer.layerName}":\n      ${newWStr}`);
    }

    return lines.join('\n');
  }

  getLatest(): InspectorEpochLog | null {
    return this.logs.length > 0 ? this.logs[this.logs.length - 1] : null;
  }

  getSnapshots(): InspectorEpochLog[] {
    return this.logs;
  }

  getFormattedLogs(): string {
    if (this.logs.length === 0) return '';
    return this.logs.map(l => l.formattedSummary).join('\n\n');
  }
}
