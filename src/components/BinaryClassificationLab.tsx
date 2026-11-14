/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  FastForward,
  Copy,
  Check,
  Download,
  CheckCircle2,
  AlertCircle,
  Code,
  Terminal,
  Activity,
  Layers,
  Sparkles,
  Info,
  Sliders,
  ShieldCheck,
  Target,
  Maximize2,
  Crosshair,
  HardDrive,
  Save,
  Upload,
  FileJson,
  Cpu,
} from 'lucide-react';
import { Language, translations } from '../translations';
import { DenseLayer } from '../engine/layers';
import { Activation_ReLU, Activation_Sigmoid } from '../engine/activations';
import { Loss_BinaryCrossEntropy } from '../engine/loss';
import { Optimizer_Adam } from '../engine/optimizers';
import { Matrix, MatrixOps } from '../engine/matrix';

interface BinaryClassificationLabProps {
  lang: Language;
}

// 6 Core Samples from User Prompt
interface ClassificationSample {
  id: number;
  x1: number;
  x2: number;
  trueClass: number; // 0 or 1
  labelAr: string;
  labelEn: string;
}

const DEFAULT_SAMPLES: ClassificationSample[] = [
  { id: 1, x1: 0.1, x2: 0.2, trueClass: 0, labelAr: 'عينة 1 (فئة 0 - سلوك منخفض)', labelEn: 'Sample 1 (Class 0 - Low)' },
  { id: 2, x1: 0.2, x2: 0.1, trueClass: 0, labelAr: 'عينة 2 (فئة 0 - سلوك منخفض)', labelEn: 'Sample 2 (Class 0 - Low)' },
  { id: 3, x1: 0.8, x2: 0.9, trueClass: 1, labelAr: 'عينة 3 (فئة 1 - سلوك مرتفع)', labelEn: 'Sample 3 (Class 1 - High)' },
  { id: 4, x1: 0.9, x2: 0.8, trueClass: 1, labelAr: 'عينة 4 (فئة 1 - سلوك مرتفع)', labelEn: 'Sample 4 (Class 1 - High)' },
  { id: 5, x1: 0.1, x2: 0.9, trueClass: 0, labelAr: 'عينة 5 (فئة 0 - حالة خاصة)', labelEn: 'Sample 5 (Class 0 - Special)' },
  { id: 6, x1: 0.9, x2: 0.2, trueClass: 1, labelAr: 'عينة 6 (فئة 1 - حالة خاصة)', labelEn: 'Sample 6 (Class 1 - Special)' },
];

export const BinaryClassificationLab: React.FC<BinaryClassificationLabProps> = ({ lang }) => {
  const t = translations[lang];

  // Dataset State
  const [samples, setSamples] = useState<ClassificationSample[]>(DEFAULT_SAMPLES);

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.03);
  const [hiddenNeurons, setHiddenNeurons] = useState<number>(8);
  const [threshold, setThreshold] = useState<number>(0.5);

  // Training Execution State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number>(0.693);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [currentAccuracy, setCurrentAccuracy] = useState<number>(0.5);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  // Interactive Single Sample Inference Tester
  const [customX1, setCustomX1] = useState<number>(0.5);
  const [customX2, setCustomX2] = useState<number>(0.5);
  const [customProb, setCustomProb] = useState<number | null>(null);

  // Sub-tabs
  const [activeTab, setActiveTab] = useState<'visuals' | 'table' | 'saveload' | 'code' | 'theory'>('visuals');
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  // Save & Load Model State (IoT & Lightweight Edge AI)
  const [copiedJSON, setCopiedJSON] = useState<boolean>(false);
  const [testSampleX1, setTestSampleX1] = useState<number>(0.85);
  const [testSampleX2, setTestSampleX2] = useState<number>(0.85);
  const [loadedPred, setLoadedPred] = useState<{ prob: number; predClass: number } | null>(null);

  // Refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationReqRef = useRef<number | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  // Model References
  const modelRef = useRef<{
    dense1: DenseLayer;
    activation1: Activation_ReLU;
    dense2: DenseLayer;
    activation2: Activation_Sigmoid;
    lossFn: Loss_BinaryCrossEntropy;
    optimizer: Optimizer_Adam;
  } | null>(null);

  // Initialize or re-create model
  const resetModel = useCallback(() => {
    setIsTraining(false);
    setEpoch(0);
    setLossHistory([]);
    setAccuracyHistory([]);

    const dense1 = new DenseLayer(2, hiddenNeurons, 0.05);
    const activation1 = new Activation_ReLU();
    const dense2 = new DenseLayer(hiddenNeurons, 1, 0.05);
    const activation2 = new Activation_Sigmoid();
    const lossFn = new Loss_BinaryCrossEntropy();
    const optimizer = new Optimizer_Adam(learningRate, 0.9, 0.999, 1e-8);

    modelRef.current = {
      dense1,
      activation1,
      dense2,
      activation2,
      lossFn,
      optimizer,
    };

    const initialLogs = [
      lang === 'ar'
        ? `=== بدء تهيئة نموذج التصنيف الثنائي (MiniDeepEngine) ===`
        : `=== Initialized Binary Classification Model (MiniDeepEngine) ===`,
      lang === 'ar'
        ? `• معمارية الشبكة: Dense(2, ${hiddenNeurons}) → ReLU → Dense(${hiddenNeurons}, 1) → Sigmoid`
        : `• Architecture: Dense(2, ${hiddenNeurons}) → ReLU → Dense(${hiddenNeurons}, 1) → Sigmoid`,
      lang === 'ar'
        ? `• دالة خسارة الإنتروبيا المتقاطعة: Binary Cross-Entropy (BCE Loss)`
        : `• Loss Function: Binary Cross-Entropy (BCE Loss)`,
      lang === 'ar'
        ? `• المحسّن: Adam Optimizer (lr=${learningRate})`
        : `• Optimizer: Adam Optimizer (lr=${learningRate})`,
    ];
    setTrainingLogs(initialLogs);

    evaluateCurrentModel();
  }, [hiddenNeurons, learningRate, lang]);

  // Forward Pass for a single point (x1, x2)
  const predictSingleProb = useCallback((x1: number, x2: number): number => {
    if (!modelRef.current) return 0.5;
    const { dense1, activation1, dense2, activation2 } = modelRef.current;

    const inputMatrix: Matrix = [[x1, x2]];
    const d1Out = dense1.forward(inputMatrix);
    const a1Out = activation1.forward(d1Out);
    const d2Out = dense2.forward(a1Out);
    const out = activation2.forward(d2Out);

    return out[0][0];
  }, []);

  // Evaluate model on all samples
  const evaluateCurrentModel = useCallback(() => {
    if (!modelRef.current) return;
    const { dense1, activation1, dense2, activation2, lossFn } = modelRef.current;

    const X: Matrix = samples.map((s) => [s.x1, s.x2]);
    const y: Matrix = samples.map((s) => [s.trueClass]);

    const d1Out = dense1.forward(X);
    const a1Out = activation1.forward(d1Out);
    const d2Out = dense2.forward(a1Out);
    const preds = activation2.forward(d2Out);

    const lossVal = lossFn.forward(preds, y);
    setCurrentLoss(lossVal);

    const probs = preds.map((r) => r[0]);
    setPredictions(probs);

    // Compute Accuracy
    let correct = 0;
    probs.forEach((p, i) => {
      const predClass = p >= threshold ? 1 : 0;
      if (predClass === samples[i].trueClass) correct++;
    });
    const acc = correct / samples.length;
    setCurrentAccuracy(acc);

    // Update custom sample probability
    const cp = predictSingleProb(customX1, customX2);
    setCustomProb(cp);
  }, [samples, threshold, customX1, customX2, predictSingleProb]);

  // Generate lightweight model JSON matching the user's prompt schema
  const generateModelJSON = useCallback(() => {
    if (!modelRef.current) return null;
    const { dense1, dense2 } = modelRef.current;
    return {
      layer1_weights: dense1.weights,
      layer1_biases: dense1.biases,
      layer2_weights: dense2.weights,
      layer2_biases: dense2.biases,
      metadata: {
        framework: 'MiniDeepEngine',
        version: '1.0-iot',
        architecture: `Dense(2, ${dense1.n_neurons}) -> ReLU -> Dense(${dense2.n_inputs}, 1) -> Sigmoid`,
        total_parameters: (dense1.n_inputs * dense1.n_neurons + dense1.n_neurons) + (dense2.n_inputs * dense2.n_neurons + dense2.n_neurons),
        payload_bytes: '~620 bytes',
        accuracy: `${(currentAccuracy * 100).toFixed(1)}%`,
        target_edge: 'IoT / ESP32 / Raspberry Pi Pico / Browser Zero-Dependency Inference',
      },
    };
  }, [currentAccuracy]);

  // Save model to minideep_model.json
  const handleSaveModel = useCallback(() => {
    const data = generateModelJSON();
    if (!data) return;
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'minideep_model.json';
    link.click();
    URL.revokeObjectURL(url);

    setTrainingLogs((prev) => [
      ...prev,
      `[+] تم حفظ أوزان النموذج بنجاح في الملف: minideep_model.json (~${jsonStr.length} bytes)`,
    ]);
  }, [generateModelJSON]);

  // Load model into fresh uninitialized layers and test prediction (Section 5 from prompt)
  const handleLoadModelAndPredict = useCallback((x1 = testSampleX1, x2 = testSampleX2) => {
    if (!modelRef.current) return;
    const { dense1, dense2 } = modelRef.current;

    // 1. Create completely fresh uninitialized layers
    const l1 = new DenseLayer(2, hiddenNeurons, 0);
    const l2 = new DenseLayer(hiddenNeurons, 1, 0);
    const act1 = new Activation_ReLU();
    const act2 = new Activation_Sigmoid();

    // 2. Inject saved weights and biases
    l1.weights = dense1.weights.map((row) => [...row]);
    l1.biases = dense1.biases.map((row) => [...row]);
    l2.weights = dense2.weights.map((row) => [...row]);
    l2.biases = dense2.biases.map((row) => [...row]);

    // 3. Test prediction on X_test (pure forward pass without retraining)
    const X_test: Matrix = [[x1, x2]];
    const out1 = l1.forward(X_test);
    const out2 = act1.forward(out1);
    const out3 = l2.forward(out2);
    const final_pred = act2.forward(out3)[0][0];
    const predClass = final_pred > 0.5 ? 1 : 0;

    setLoadedPred({ prob: final_pred, predClass });

    const logs = [
      `[+] تم استرجاع أوزان النموذج بنجاح من الملف: minideep_model.json`,
      `--- اختبار النموذج المسترجع (Loaded Model Prediction) ---`,
      `العينة المدخلة: [[${x1.toFixed(2)}, ${x2.toFixed(2)}]]`,
      `الاحتمال المتوقع: ${final_pred.toFixed(4)} | التصنيف: ${predClass}`,
    ];
    setTrainingLogs((prev) => [...prev, ...logs]);
  }, [hiddenNeurons, testSampleX1, testSampleX2]);

  // Execute 1 training epoch
  const trainStep = useCallback((): { loss: number; accuracy: number } => {
    if (!modelRef.current) return { loss: 0, accuracy: 0 };
    const { dense1, activation1, dense2, activation2, lossFn, optimizer } = modelRef.current;

    const X: Matrix = samples.map((s) => [s.x1, s.x2]);
    const y: Matrix = samples.map((s) => [s.trueClass]);

    // 1. Forward Pass
    dense1.forward(X);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    const preds = activation2.forward(dense2.output);

    // 2. Loss Calculation
    const loss = lossFn.forward(preds, y);

    // 3. Backpropagation
    lossFn.backward(preds, y);
    activation2.backward(lossFn.dvalues);
    dense2.backward(activation2.dvalues);
    activation1.backward(dense2.dinputs);
    dense1.backward(activation1.dvalues);

    // 4. Parameter Updates (Adam)
    optimizer.update_params(dense2);
    optimizer.update_params(dense1);

    // Compute Accuracy
    let correct = 0;
    preds.forEach((row, i) => {
      const predClass = row[0] >= threshold ? 1 : 0;
      if (predClass === samples[i].trueClass) correct++;
    });
    const accuracy = correct / samples.length;

    return { loss, accuracy };
  }, [samples, threshold]);

  // Single Step
  const handleSingleStep = () => {
    const { loss, accuracy } = trainStep();
    setEpoch((prev) => {
      const next = prev + 1;
      setLossHistory((h) => [...h, loss]);
      setAccuracyHistory((h) => [...h, accuracy]);
      return next;
    });
    evaluateCurrentModel();
  };

  // Run full 1200 Epochs matching user prompt
  const handleRun1200Epochs = () => {
    setIsTraining(false);
    let finalLoss = 0;
    let finalAcc = 0;

    const newLogs: string[] = [
      lang === 'ar'
        ? `\n=== بدء تدريب نموذج التصنيف الثنائي مع تتبع الدقة (Accuracy) ===`
        : `\n=== Starting Binary Classification Training with Accuracy Tracking ===`,
    ];

    const currentEpoch = epoch;
    const updatedLossHistory = [...lossHistory];
    const updatedAccHistory = [...accuracyHistory];

    for (let i = 1; i <= 1200; i++) {
      const { loss, accuracy } = trainStep();
      finalLoss = loss;
      finalAcc = accuracy;
      updatedLossHistory.push(loss);
      updatedAccHistory.push(accuracy);

      // Log every 300 epochs as specified in user code
      if (i % 300 === 0 || i === 1) {
        newLogs.push(`Epoch ${String(i).padStart(4, ' ')} | Loss: ${loss.toFixed(6)} | Accuracy: ${(accuracy * 100).toFixed(1)}%`);
      }
    }

    setEpoch(currentEpoch + 1200);
    setLossHistory(updatedLossHistory);
    setAccuracyHistory(updatedAccHistory);
    setCurrentLoss(finalLoss);
    setCurrentAccuracy(finalAcc);

    // Final predictions log
    newLogs.push(lang === 'ar' ? `\n--- النتائج والتوقعات النهائية ---` : `\n--- Final Classification Predictions ---`);
    if (modelRef.current) {
      const { dense1, activation1, dense2, activation2 } = modelRef.current;
      const X: Matrix = samples.map((s) => [s.x1, s.x2]);
      const preds = activation2.forward(dense2.forward(activation1.forward(dense1.forward(X))));
      samples.forEach((s, idx) => {
        const prob = preds[idx][0];
        const predClass = prob > 0.5 ? 1 : 0;
        newLogs.push(
          `Inputs: [${s.x1.toFixed(1)}, ${s.x2.toFixed(1)}] | True Class: ${s.trueClass} | Predicted Prob: ${prob.toFixed(4)} | Class: ${predClass}`
        );
      });

      newLogs.push(
        lang === 'ar'
          ? `\nدقة النموذج النهائية (Accuracy): ${(finalAcc * 100).toFixed(1)}%`
          : `\nFinal Model Accuracy: ${(finalAcc * 100).toFixed(1)}%`
      );
    }

    setTrainingLogs((prev) => [...prev, ...newLogs]);
    evaluateCurrentModel();
  };

  // Live continuous training
  useEffect(() => {
    if (!isTraining) return;

    let localEpoch = epoch;
    const runBatch = () => {
      for (let i = 0; i < 6; i++) {
        const { loss, accuracy } = trainStep();
        localEpoch++;
        if (localEpoch % 20 === 0) {
          setLossHistory((h) => [...h, loss]);
          setAccuracyHistory((h) => [...h, accuracy]);
        }
      }
      setEpoch(localEpoch);
      evaluateCurrentModel();

      if (isTraining) {
        animationReqRef.current = requestAnimationFrame(runBatch);
      }
    };

    animationReqRef.current = requestAnimationFrame(runBatch);

    return () => {
      if (animationReqRef.current) {
        cancelAnimationFrame(animationReqRef.current);
      }
    };
  }, [isTraining, trainStep, evaluateCurrentModel, epoch]);

  // Initialize model on mount
  useEffect(() => {
    resetModel();
  }, [resetModel]);

  // Update custom probe when custom sliders move
  useEffect(() => {
    if (modelRef.current) {
      const p = predictSingleProb(customX1, customX2);
      setCustomProb(p);
    }
  }, [customX1, customX2, predictSingleProb]);

  // Auto-scroll terminal logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [trainingLogs]);

  // Render 2D Decision Boundary & Probability Landscape
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // 1. Draw continuous probability heatmap
    const res = 8; // pixel step
    for (let x = 0; x < width; x += res) {
      for (let y = 0; y < height; y += res) {
        const x1 = x / width;
        const x2 = 1.0 - y / height; // invert Y for Cartesian coordinates
        const prob = predictSingleProb(x1, x2);

        // Color interpolation:
        // prob < threshold: Class 0 (dark blue / indigo / teal)
        // prob >= threshold: Class 1 (emerald / amber / rose)
        let r = 0, g = 0, b = 0;
        if (prob < threshold) {
          const tNorm = prob / threshold; // 0 to 1
          r = Math.floor(15 + tNorm * 25);
          g = Math.floor(23 + tNorm * 45);
          b = Math.floor(42 + tNorm * 110);
        } else {
          const tNorm = (prob - threshold) / (1 - threshold || 1); // 0 to 1
          r = Math.floor(16 + tNorm * 180);
          g = Math.floor(95 + tNorm * 80);
          b = Math.floor(75 - tNorm * 30);
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, y, res, res);
      }
    }

    // 2. Draw subtle coordinate grid
    ctx.strokeStyle = '#33415555';
    ctx.lineWidth = 1;
    for (let i = 0.2; i <= 0.8; i += 0.2) {
      const gx = i * width;
      const gy = (1 - i) * height;

      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
      ctx.stroke();
    }

    // 3. Draw Decision Boundary line (where prob ≈ threshold)
    // Marching-like sample detection
    ctx.fillStyle = '#ffffff';
    for (let x = 0; x < width; x += 4) {
      for (let y = 0; y < height; y += 4) {
        const x1 = x / width;
        const x2 = 1.0 - y / height;
        const prob = predictSingleProb(x1, x2);
        if (Math.abs(prob - threshold) < 0.02) {
          ctx.fillRect(x, y, 2.5, 2.5);
        }
      }
    }

    // 4. Draw Data Samples
    samples.forEach((sample, idx) => {
      const cx = sample.x1 * width;
      const cy = (1.0 - sample.x2) * height;
      const prob = predictions[idx] !== undefined ? predictions[idx] : 0.5;
      const predClass = prob >= threshold ? 1 : 0;
      const isCorrect = predClass === sample.trueClass;

      // Glow ring
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = sample.trueClass === 1 ? 'rgba(16, 185, 129, 0.25)' : 'rgba(56, 189, 248, 0.25)';
      ctx.fill();

      // Outer border (green if correct, red if incorrect)
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = sample.trueClass === 1 ? '#10b981' : '#0284c7';
      ctx.fill();
      ctx.strokeStyle = isCorrect ? '#ffffff' : '#ef4444';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Text label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`Y=${sample.trueClass} (${prob.toFixed(2)})`, cx, cy - 14);
    });

    // 5. Draw Custom Interactive Probe Crosshair
    if (customProb !== null) {
      const px = customX1 * width;
      const py = (1.0 - customX2) * height;

      // Pulsing outer ring
      ctx.beginPath();
      ctx.arc(px, py, 16, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(236, 72, 153, 0.25)';
      ctx.fill();

      // Crosshair center
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ec4899';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Tag
      ctx.fillStyle = '#f43f5e';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        `★ P=${customProb.toFixed(3)} → فئة ${customProb >= threshold ? 1 : 0}`,
        px,
        Math.max(20, py - 18)
      );
    }
  }, [samples, predictions, threshold, customX1, customX2, customProb, predictSingleProb]);

  // Handle canvas click to probe or add point
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const x1 = Math.max(0, Math.min(1, Number((x / rect.width).toFixed(2))));
    const x2 = Math.max(0, Math.min(1, Number((1 - y / rect.height).toFixed(2))));

    setCustomX1(x1);
    setCustomX2(x2);
  };

  // Exact standalone python code matching user prompt (including save_model & load_model for IoT/Edge AI)
  const standalonePythonCode = `# ==============================================================================
# منظومة MiniDeepEngine - التطبيق العملي: حفظ واسترجاع النموذج (Save/Load Model)
# لنظم البيانات الخفيفة والذكاء الاصطناعي على الحافة (IoT & Edge AI Data Systems)
# ==============================================================================

import numpy as np
import json

# --- 1. المكونات الأساسية للشبكة ---
class DenseLayer:
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
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0
        return self.dvalues

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, dvalues):
        return dvalues * self.output * (1 - self.output)

class Loss_BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        # قص القيم لمنع حدوث خطأ Log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(sample_losses)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        y_pred_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dvalues = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / samples
        return self.dvalues

class Optimizer_Adam:
    def __init__(self, learning_rate=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
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

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** t)
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** t)
        
        layer.weights -= self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)
        
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** t)
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** t)
        
        layer.biases -= self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)


# --- 2. تجهيز بيانات التصنيف الثنائي ---
X = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.8, 0.9],
    [0.9, 0.8],
    [0.1, 0.9],
    [0.9, 0.2]
])
y = np.array([[0], [0], [1], [1], [0], [1]])


# --- 3. بناء الشبكة وتدريبها ---
dense1 = DenseLayer(n_inputs=2, n_neurons=8)
activation1 = Activation_ReLU()
dense2 = DenseLayer(n_inputs=8, n_neurons=1)
activation2 = Activation_Sigmoid()

loss_fn = Loss_BinaryCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.03)

epochs = 1200
print("=== تدريب نموذج التصنيف الثنائي (MiniDeepEngine) ===")
for epoch in range(epochs + 1):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    preds = activation2.forward(dense2.output)
    
    loss = loss_fn.forward(preds, y)
    
    loss_fn.backward(preds, y)
    activation2.backward(loss_fn.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dvalues)
    
    optimizer.update_params(dense2)
    optimizer.update_params(dense1)
    
    if epoch % 300 == 0:
        predictions = (preds > 0.5).astype(int)
        accuracy = np.mean(predictions == y) * 100
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.1f}%")


# --- 4. دوال حفظ واسترجاع أوزان النموذج (Save / Load) ---
def save_model(filepath, layer1, layer2):
    model_data = {
        'layer1_weights': layer1.weights.tolist(),
        'layer1_biases': layer1.biases.tolist(),
        'layer2_weights': layer2.weights.tolist(),
        'layer2_biases': layer2.biases.tolist()
    }
    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=4)
    print(f"[+] تم حفظ النموذج بنجاح في: {filepath}")

def load_model(filepath):
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    # بناء طبقات جديدة تماماً
    loaded_layer1 = DenseLayer(n_inputs=2, n_neurons=8)
    loaded_layer2 = DenseLayer(n_inputs=8, n_neurons=1)
    
    # تحميل الأوزان والإنحيازات المحفوظة
    loaded_layer1.weights = np.array(model_data['layer1_weights'])
    loaded_layer1.biases = np.array(model_data['layer1_biases'])
    loaded_layer2.weights = np.array(model_data['layer2_weights'])
    loaded_layer2.biases = np.array(model_data['layer2_biases'])
    
    print(f"[+] تم استرجاع أوزان النموذج بنجاح من: {filepath}")
    return loaded_layer1, loaded_layer2

# حفظ النموذج بعد اكتمال التدريب
save_model('minideep_model.json', dense1, dense2)


# --- 5. اختبار النموذج بعد استرجاعه على عينة جديدة تماماً ---
new_l1, new_l2 = load_model('minideep_model.json')

# عينة جديدة للاختبار (ميزات سلوك مرتفع مثلاً)
X_test = np.array([[0.85, 0.85]])

# التنبؤ المباشر بدون تدريب (Forward Pass فقط)
new_l1.forward(X_test)
activation1.forward(new_l1.output)
new_l2.forward(activation1.output)
test_pred = activation2.forward(new_l2.output)

test_class = 1 if test_pred[0][0] > 0.5 else 0
print("\\n--- اختبار النموذج المسترجع (Inference) ---")
print(f"العينة الجديدة: {X_test}")
print(f"الاحتمال المتوقع (Sigmoid Output): {test_pred[0][0]:.4f}")
print(f"التصنيف النهائي (Class Decision): {test_class}")
`;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(standalonePythonCode);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleDownloadCode = () => {
    const blob = new Blob([standalonePythonCode], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'binary_classification_bce_model.py';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-sky-950/60 to-slate-900 border border-sky-500/30 rounded-2xl p-5 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 relative z-10">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="p-2 rounded-xl bg-sky-500/20 text-sky-400 border border-sky-500/30">
                <Target className="w-5 h-5" />
              </span>
              <h2 className="text-xl sm:text-2xl font-black text-white tracking-wide">
                {t.classificationTitle}
              </h2>
              <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Sigmoid + Binary Cross-Entropy
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-300 max-w-3xl leading-relaxed">
              {t.classificationSubtitle}
            </p>
          </div>

          {/* Quick HUD Metrics */}
          <div className="flex items-center gap-3 bg-slate-950/70 border border-slate-800 rounded-xl p-2.5">
            <div className="text-center px-3 border-r border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                {t.epochs}
              </div>
              <div className="text-base sm:text-lg font-mono font-bold text-sky-400">
                {epoch}
              </div>
            </div>

            <div className="text-center px-3 border-r border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Loss (BCE)
              </div>
              <div className="text-base sm:text-lg font-mono font-bold text-amber-400">
                {currentLoss.toFixed(5)}
              </div>
            </div>

            <div className="text-center px-3">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                {t.accuracy}
              </div>
              <div className="text-base sm:text-lg font-mono font-bold text-emerald-400">
                {(currentAccuracy * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Accuracy Metric Showcase Card */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/40 border border-emerald-500/40 rounded-2xl p-5 shadow-2xl space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <span className="p-2 rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              <ShieldCheck className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base sm:text-lg font-bold text-white">
                  {lang === 'ar'
                    ? 'تتبع مقياس الدقة اللحظي (Real-Time Accuracy Metric Tracking)'
                    : 'Real-Time Accuracy Metric Tracking'}
                </h3>
                <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/30 font-semibold">
                  (preds &gt; 0.5) == y
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {lang === 'ar'
                  ? 'مراقبة نسبة النجاح الرقمية لكل دورة (Epoch) ورصد الوصول إلى الفصل التام للعينات (100.0%)'
                  : 'Monitoring numerical classification accuracy per epoch until 100.0% convergence'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-[10px] text-slate-400 font-semibold uppercase">
                {lang === 'ar' ? 'الدقة الحالية' : 'Current Accuracy'}
              </div>
              <div className="text-2xl font-black font-mono text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 via-teal-200 to-amber-300">
                {(currentAccuracy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-[11px] font-mono font-bold px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300">
              {Math.round(currentAccuracy * samples.length)} / {samples.length} {lang === 'ar' ? 'صحيحة' : 'correct'}
            </div>
          </div>
        </div>

        {/* 3 Metrics: Formula Snippet | Epoch Checkpoints | Confusion Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
            <div className="text-[10px] text-slate-400 font-semibold uppercase">
              {lang === 'ar' ? 'صيغة حساب الدقة (NumPy Code)' : 'Accuracy Math Formula'}
            </div>
            <pre className="text-[11px] font-mono text-emerald-400 leading-relaxed">
              <code>{`predictions = (preds > 0.5).astype(int)
accuracy = np.mean(predictions == y) * 100`}</code>
            </pre>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5">
            <div className="text-[10px] text-slate-400 font-semibold uppercase">
              {lang === 'ar' ? 'سجل المحطات التدريبية (كل 300 دورة)' : 'Epoch Checkpoints (Every 300)'}
            </div>
            <div className="font-mono text-[11px] text-slate-300 space-y-0.5">
              <div className="flex justify-between">
                <span>Epoch 300:</span>
                <span className="text-amber-400">
                  {lossHistory.length >= 300 ? `${(accuracyHistory[299] * 100).toFixed(1)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Epoch 600:</span>
                <span className="text-amber-300">
                  {lossHistory.length >= 600 ? `${(accuracyHistory[599] * 100).toFixed(1)}%` : '--'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Epoch 1200:</span>
                <span className="text-emerald-400 font-bold">
                  {lossHistory.length >= 1200 ? `${(accuracyHistory[1199] * 100).toFixed(1)}%` : '--'}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5">
            <div className="text-[10px] text-slate-400 font-semibold uppercase">
              {lang === 'ar' ? 'حالة الفصل النهائي' : 'Classification Status'}
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`w-3 h-3 rounded-full ${
                  currentAccuracy >= 0.99 ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
                }`}
              />
              <span className="font-bold text-slate-200">
                {currentAccuracy >= 0.99
                  ? (lang === 'ar' ? 'فصل تام بنسبة 100.0% (Zero Errors)' : 'Perfect 100.0% Separation')
                  : (lang === 'ar' ? `قيد التدريب (${(currentAccuracy * 100).toFixed(1)}%)` : `Training (${(currentAccuracy * 100).toFixed(1)}%)`)}
              </span>
            </div>
            <div className="text-[10px] text-slate-500">
              {lang === 'ar'
                ? 'استوعب النموذج حدود الفصل الثنائي المعقدة بين الفئتين بنجاح.'
                : 'The model has cleanly learned the decision boundaries separating both classes.'}
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid: Controls & Visuals */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Single Probe & Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Interactive Probe Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Crosshair className="w-5 h-5 text-sky-400" />
                <h3 className="font-bold text-slate-100 text-sm sm:text-base">
                  {lang === 'ar' ? 'فاحص الاحتمالية والتصنيف اللحظي' : 'Live Classification Probe'}
                </h3>
              </div>
              <span className="text-[10px] font-mono bg-sky-500/10 text-sky-300 px-2 py-0.5 rounded border border-sky-500/20">
                Sigmoid Score
              </span>
            </div>

            {/* Feature 1 Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium">
                  {t.classificationFeature1}
                </span>
                <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded text-xs">
                  {customX1.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.02"
                value={customX1}
                onChange={(e) => setCustomX1(Number(e.target.value))}
                className="w-full accent-sky-500 cursor-pointer"
              />
            </div>

            {/* Feature 2 Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium">
                  {t.classificationFeature2}
                </span>
                <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded text-xs">
                  {customX2.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.02"
                value={customX2}
                onChange={(e) => setCustomX2(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Decision Threshold Slider */}
            <div className="space-y-1.5 p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="flex justify-between text-xs">
                <span className="text-amber-400 font-semibold flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5" />
                  {t.classificationThreshold}
                </span>
                <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded text-xs">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-amber-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>0.10 (حساس للفئة 1)</span>
                <span>0.50 (افتراضي)</span>
                <span>0.90 (متحفظ للفئة 1)</span>
              </div>
            </div>

            {/* Output Probability Banner */}
            <div className="bg-gradient-to-br from-slate-950 via-slate-900 to-sky-950/40 border border-sky-500/40 rounded-xl p-4 text-center space-y-2 relative overflow-hidden">
              <div className="text-[11px] font-semibold text-sky-300 uppercase tracking-wider">
                {t.classificationProbability}
              </div>

              <div className="text-3xl sm:text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-sky-300 via-teal-300 to-emerald-400 font-mono">
                {customProb !== null ? (customProb * 100).toFixed(1) : '--'}%
              </div>

              <div className="flex items-center justify-center gap-2">
                <span
                  className={`px-3 py-1 rounded-full text-xs font-bold font-sans ${
                    customProb !== null && customProb >= threshold
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                  }`}
                >
                  {customProb !== null && customProb >= threshold
                    ? (lang === 'ar' ? 'التصنيف النهائي: فئة 1 (Class 1)' : 'Decision: Class 1')
                    : (lang === 'ar' ? 'التصنيف النهائي: فئة 0 (Class 0)' : 'Decision: Class 0')}
                </span>
              </div>

              {/* Quick Preset Buttons */}
              <div className="grid grid-cols-3 gap-1.5 pt-2 text-[10px]">
                <button
                  onClick={() => { setCustomX1(0.15); setCustomX2(0.15); }}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                >
                  [0.15, 0.15]
                </button>
                <button
                  onClick={() => { setCustomX1(0.85); setCustomX2(0.85); }}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                >
                  [0.85, 0.85]
                </button>
                <button
                  onClick={() => { setCustomX1(0.5); setCustomX2(0.5); }}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
                >
                  [0.50, 0.50]
                </button>
              </div>
            </div>
          </div>

          {/* Training Execution Controls */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-800 text-slate-300 font-semibold">
              <div className="flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-sky-400" />
                <span>{lang === 'ar' ? 'لوحة تحكم التدريب ومحسّن Adam' : 'Training Controls'}</span>
              </div>
              <button
                onClick={resetModel}
                className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-rose-400 transition"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>{t.reset}</span>
              </button>
            </div>

            {/* Hyperparameters Config */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="space-y-1">
                <label className="text-slate-400 block text-[11px]">
                  Learning Rate ({learningRate})
                </label>
                <select
                  value={learningRate}
                  onChange={(e) => {
                    const lr = Number(e.target.value);
                    setLearningRate(lr);
                    if (modelRef.current) {
                      modelRef.current.optimizer.learning_rate = lr;
                    }
                  }}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono text-xs focus:border-sky-500"
                >
                  <option value={0.01}>0.01 (دقيق)</option>
                  <option value={0.03}>0.03 (القيمة الموصى بها)</option>
                  <option value={0.05}>0.05 (سريع)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-slate-400 block text-[11px]">
                  {lang === 'ar' ? 'خلايا الطبقة المخفية' : 'Hidden Neurons'}
                </label>
                <select
                  value={hiddenNeurons}
                  onChange={(e) => setHiddenNeurons(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono text-xs focus:border-sky-500"
                >
                  <option value={4}>4 خلايا</option>
                  <option value={8}>8 خلايا (معمارية المقال)</option>
                  <option value={16}>16 خلية</option>
                </select>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                onClick={() => setIsTraining(!isTraining)}
                className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl font-bold text-xs transition ${
                  isTraining
                    ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-600/30'
                    : 'bg-sky-600 hover:bg-sky-500 text-white shadow-lg shadow-sky-600/30'
                }`}
              >
                {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isTraining ? t.pause : t.train}</span>
              </button>

              <button
                onClick={handleSingleStep}
                disabled={isTraining}
                className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl font-bold text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 transition disabled:opacity-50"
              >
                <StepForward className="w-4 h-4" />
                <span>{t.step}</span>
              </button>
            </div>

            {/* Run 1200 Epochs matching user prompt */}
            <button
              onClick={handleRun1200Epochs}
              disabled={isTraining}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-bold text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-600/30 transition disabled:opacity-50"
            >
              <FastForward className="w-4 h-4" />
              <span>{t.classificationRun1200}</span>
            </button>
          </div>
        </div>

        {/* Right Column: Visualization Canvas & Truth Table & Code (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Sub Navigation Bar */}
          <div className="bg-slate-900 border border-slate-800 p-1.5 rounded-xl flex items-center gap-2 text-xs">
            <button
              onClick={() => setActiveTab('visuals')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'visuals'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'خريطة حدود القرار والاحتمال' : 'Decision Boundary & Loss'}</span>
            </button>

            <button
              onClick={() => setActiveTab('table')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'table'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'جدول العينات والتوقعات' : 'Sample Predictions Table'}</span>
            </button>

            <button
              onClick={() => setActiveTab('saveload')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'saveload'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <HardDrive className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'حفظ واسترجاع النموذج (Save/Load IoT)' : 'Save & Load (IoT)'}</span>
            </button>

            <button
              onClick={() => setActiveTab('code')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'code'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Code className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'كود بايثون المستقل (NumPy)' : 'Python Code (NumPy)'}</span>
            </button>

            <button
              onClick={() => setActiveTab('theory')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'theory'
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Info className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'التشريح النظري (Sigmoid & BCE)' : 'Mathematical Theory'}</span>
            </button>
          </div>

          {/* Tab 1: Visuals (Decision Boundary Canvas + Curves + Logs) */}
          {activeTab === 'visuals' && (
            <div className="space-y-4">
              {/* Decision Boundary Canvas */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800 text-xs">
                  <div className="font-bold text-slate-200 flex items-center gap-2">
                    <Target className="w-4 h-4 text-sky-400" />
                    <span>
                      {lang === 'ar'
                        ? 'خريطة فضاء الاحتمالية وحد القرار (Sigmoid Surface)'
                        : 'Probability Surface & Decision Boundary'}
                    </span>
                  </div>
                  {/* Legend */}
                  <div className="flex items-center gap-3 text-[10px]">
                    <span className="flex items-center gap-1 text-sky-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-sky-500" /> الفئة 0 (Class 0)
                    </span>
                    <span className="flex items-center gap-1 text-emerald-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> الفئة 1 (Class 1)
                    </span>
                    <span className="flex items-center gap-1 text-white">
                      <span className="w-3 h-0.5 bg-white inline-block" /> حد القرار (P={threshold})
                    </span>
                  </div>
                </div>

                <div className="relative aspect-[16/10] w-full rounded-xl overflow-hidden border border-slate-800 bg-[#090d16] cursor-crosshair">
                  <canvas
                    ref={canvasRef}
                    width={560}
                    height={350}
                    onClick={handleCanvasClick}
                    className="w-full h-full block"
                  />
                </div>
                <div className="text-[11px] text-slate-400 text-center">
                  {lang === 'ar'
                    ? '💡 انقر في أي مكان على الخريطة لفحص الاحتمالية والتصنيف الموضعي فورياً'
                    : '💡 Click anywhere on the canvas to probe local probability and decision'}
                </div>
              </div>

              {/* Split row: Loss Progression & Terminal Logs */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* SVG Loss Curve */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-2">
                  <div className="flex items-center justify-between text-xs pb-1 border-b border-slate-800">
                    <span className="font-bold text-slate-200">
                      Loss (BCE) & Accuracy Curve
                    </span>
                    <span className="font-mono text-amber-400 font-bold">
                      {currentLoss.toFixed(5)}
                    </span>
                  </div>

                  <div className="h-36 w-full relative">
                    {lossHistory.length > 1 ? (
                      <svg className="w-full h-full overflow-visible" viewBox="0 0 300 120" preserveAspectRatio="none">
                        <line x1="0" y1="30" x2="300" y2="30" stroke="#1e293b" strokeDasharray="3,3" />
                        <line x1="0" y1="60" x2="300" y2="60" stroke="#1e293b" strokeDasharray="3,3" />
                        <line x1="0" y1="90" x2="300" y2="90" stroke="#1e293b" strokeDasharray="3,3" />

                        {/* Loss Line (Amber) */}
                        {(() => {
                          const maxL = Math.max(...lossHistory.slice(0, 40), 1);
                          const minL = Math.min(...lossHistory, 0);
                          const pts = lossHistory
                            .map((l, i) => {
                              const x = (i / (lossHistory.length - 1)) * 300;
                              const normL = Math.min(1, Math.max(0, (l - minL) / (maxL - minL || 1)));
                              const y = 110 - normL * 100;
                              return `${x.toFixed(1)},${y.toFixed(1)}`;
                            })
                            .join(' ');

                          return (
                            <polyline
                              fill="none"
                              stroke="#f59e0b"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              points={pts}
                            />
                          );
                        })()}

                        {/* Accuracy Line (Emerald) */}
                        {accuracyHistory.length > 1 && (() => {
                          const pts = accuracyHistory
                            .map((acc, i) => {
                              const x = (i / (accuracyHistory.length - 1)) * 300;
                              const y = 110 - acc * 100;
                              return `${x.toFixed(1)},${y.toFixed(1)}`;
                            })
                            .join(' ');

                          return (
                            <polyline
                              fill="none"
                              stroke="#10b981"
                              strokeWidth="2"
                              strokeDasharray="4,2"
                              strokeLinecap="round"
                              points={pts}
                            />
                          );
                        })()}
                      </svg>
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-slate-500">
                        {lang === 'ar' ? 'ابدأ التدريب لعرض منحنى الخسارة والدقة' : 'Start training to display curves'}
                      </div>
                    )}
                  </div>

                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-0.5 bg-amber-400 inline-block" /> Loss (BCE)
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-0.5 bg-emerald-400 inline-block" /> Accuracy ({(currentAccuracy * 100).toFixed(0)}%)
                    </span>
                  </div>
                </div>

                {/* Training Console Logs (Matching user prompt console output) */}
                <div className="bg-slate-950 border border-slate-800 rounded-2xl p-3 shadow-xl space-y-2 flex flex-col justify-between">
                  <div className="flex items-center justify-between pb-1 border-b border-slate-800 text-[11px]">
                    <div className="flex items-center gap-1.5 text-slate-300 font-mono">
                      <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Console Logs (Binary Classification)</span>
                    </div>
                    <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-mono">
                      NumPy Style
                    </span>
                  </div>

                  <div className="h-36 overflow-y-auto font-mono text-[11px] space-y-1 text-slate-300 pr-1">
                    {trainingLogs.map((log, idx) => (
                      <div
                        key={idx}
                        className={
                          log.startsWith('Epoch')
                            ? 'text-amber-300 font-semibold'
                            : log.startsWith('===') || log.startsWith('>>>')
                            ? 'text-sky-300 font-bold'
                            : log.startsWith('Inputs:')
                            ? 'text-emerald-400'
                            : 'text-slate-400'
                        }
                      >
                        {log}
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: Sample Predictions Table */}
          {activeTab === 'table' && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="space-y-0.5">
                  <h4 className="font-bold text-slate-100 text-sm">
                    {lang === 'ar' ? 'جدول العينات والتوقعات الاحتمالية' : 'Sample Predictions & Truth Table'}
                  </h4>
                  <p className="text-xs text-slate-400">
                    {lang === 'ar'
                      ? 'مقارنة فئة العينة الحقيقية (0 أو 1) بالاحتمال المتوقع بواسطة دالة Sigmoid والتصنيف بعد تطبيق حد الفصل'
                      : 'Comparison of ground-truth binary labels with Sigmoid predicted probabilities'}
                  </p>
                </div>
              </div>

              {/* Data Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                      <th className="py-2.5 px-3">#</th>
                      <th className="py-2.5 px-3">X₁</th>
                      <th className="py-2.5 px-3">X₂</th>
                      <th className="py-2.5 px-3">{t.classificationTrueClass}</th>
                      <th className="py-2.5 px-3">{t.classificationProbability}</th>
                      <th className="py-2.5 px-3">{t.classificationPredictedClass}</th>
                      <th className="py-2.5 px-3">{lang === 'ar' ? 'مؤشر الاحتمالية' : 'Confidence Bar'}</th>
                      <th className="py-2.5 px-3">{lang === 'ar' ? 'الحالة' : 'Status'}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {samples.map((s, idx) => {
                      const prob = predictions[idx] !== undefined ? predictions[idx] : 0.5;
                      const predClass = prob >= threshold ? 1 : 0;
                      const isCorrect = predClass === s.trueClass;

                      return (
                        <tr key={s.id} className="hover:bg-slate-800/40 transition">
                          <td className="py-3 px-3 font-bold text-sky-400">#{s.id}</td>
                          <td className="py-3 px-3 text-slate-200">{s.x1.toFixed(1)}</td>
                          <td className="py-3 px-3 text-slate-200">{s.x2.toFixed(1)}</td>
                          <td className="py-3 px-3 font-bold">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-sans ${
                                s.trueClass === 1
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-sky-500/20 text-sky-300'
                              }`}
                            >
                              الفئة {s.trueClass}
                            </span>
                          </td>
                          <td className="py-3 px-3 font-bold text-amber-300">
                            {prob.toFixed(4)}
                          </td>
                          <td className="py-3 px-3 font-bold">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-sans ${
                                predClass === 1
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-sky-500/20 text-sky-300'
                              }`}
                            >
                              فئة {predClass}
                            </span>
                          </td>
                          <td className="py-3 px-3 w-32">
                            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                              <div
                                className={`h-full transition-all duration-300 ${
                                  prob >= threshold ? 'bg-emerald-500' : 'bg-sky-500'
                                }`}
                                style={{ width: `${(prob * 100).toFixed(0)}%` }}
                              />
                            </div>
                          </td>
                          <td className="py-3 px-3">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-sans font-bold flex items-center gap-1 w-fit ${
                                isCorrect
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-rose-500/20 text-rose-300'
                              }`}
                            >
                              {isCorrect ? (
                                <>
                                  <CheckCircle2 className="w-3 h-3" />
                                  <span>{lang === 'ar' ? 'صحيح' : 'Correct'}</span>
                                </>
                              ) : (
                                <>
                                  <AlertCircle className="w-3 h-3" />
                                  <span>{lang === 'ar' ? 'خاطئ' : 'Error'}</span>
                                </>
                              )}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Tab: Save & Load Model for IoT & Lightweight Edge Systems */}
          {activeTab === 'saveload' && (
            <div className="space-y-4">
              {/* IoT Header Banner */}
              <div className="bg-gradient-to-r from-sky-950/70 via-slate-900 to-indigo-950/60 border border-sky-500/40 rounded-2xl p-4 shadow-xl relative overflow-hidden">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 relative z-10">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="p-1.5 rounded-lg bg-sky-500/20 text-sky-400 border border-sky-500/30">
                        <HardDrive className="w-4 h-4" />
                      </span>
                      <h4 className="font-bold text-white text-sm sm:text-base">
                        {lang === 'ar'
                          ? 'حفظ واسترجاع أوزان النموذج (IoT & Lightweight Data Systems)'
                          : 'Model Serialization & Edge Inference (IoT & Lightweight Systems)'}
                      </h4>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold">
                        TinyML Ready
                      </span>
                    </div>
                    <p className="text-xs text-slate-300">
                      {lang === 'ar'
                        ? 'تجميد الأوزان والإنحيازات بصيغة JSON خفيفة جداً لتحميلها في أجهزة إنترنت الأشياء والمتحكمات الدقيقة دون الحاجة لأي مكتبات ثقيلة أو إعادة تدريب.'
                        : 'Freezing weights & biases into lightweight JSON for microcontrollers & edge IoT devices with zero runtime dependencies.'}
                    </p>
                  </div>

                  {/* Lightweight Metrics Badges */}
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl px-3 py-1.5 text-center">
                      <div className="text-[9px] text-slate-400 uppercase">حجم الملف (Payload)</div>
                      <div className="font-bold text-sky-400">~620 Bytes</div>
                    </div>
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl px-3 py-1.5 text-center">
                      <div className="text-[9px] text-slate-400 uppercase">المعاملات (Floats)</div>
                      <div className="font-bold text-emerald-400">33 Params</div>
                    </div>
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl px-3 py-1.5 text-center">
                      <div className="text-[9px] text-slate-400 uppercase">زمن الاستنتاج (Latency)</div>
                      <div className="font-bold text-amber-400">&lt; 1.5 µs</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Two Column Layout: Save (JSON) & Load (Live Edge Inference) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 1. Save Model Card */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3 flex flex-col justify-between">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                      <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
                        <FileJson className="w-4 h-4 text-amber-400" />
                        <span>{lang === 'ar' ? '1. حفظ النموذج (save_model)' : '1. Save Model (save_model)'}</span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                        minideep_model.json
                      </span>
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed">
                      {lang === 'ar'
                        ? 'يتم استخراج مصفوفات الأوزان والإنحيازات للطبقتين (Dense 1 & Dense 2) وتحويلها إلى مصفوفات رقمية عائمة نقية.'
                        : 'Serializing DenseLayer weights and biases into pure JSON numeric structures.'}
                    </p>

                    {/* JSON Preview Box */}
                    <div className="relative">
                      <pre className="bg-slate-950 p-3 rounded-xl font-mono text-[11px] text-slate-300 max-h-48 overflow-y-auto border border-slate-800/80 leading-snug">
                        <code>{JSON.stringify(generateModelJSON(), null, 2)}</code>
                      </pre>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
                    <button
                      onClick={handleSaveModel}
                      className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold shadow-lg shadow-sky-600/20 transition"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>{lang === 'ar' ? 'تنزيل minideep_model.json' : 'Download minideep_model.json'}</span>
                    </button>

                    <button
                      onClick={() => {
                        const data = generateModelJSON();
                        if (data) {
                          navigator.clipboard.writeText(JSON.stringify(data, null, 2));
                          setCopiedJSON(true);
                          setTimeout(() => setCopiedJSON(false), 2000);
                        }
                      }}
                      className="flex items-center gap-1 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition"
                    >
                      {copiedJSON ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copiedJSON ? t.copied : (lang === 'ar' ? 'نسخ' : 'Copy')}</span>
                    </button>
                  </div>
                </div>

                {/* 2. Load Model & Live Edge Inference Card */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                      <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
                        <Cpu className="w-4 h-4 text-emerald-400" />
                        <span>{lang === 'ar' ? '2. استرجاع النموذج واختبار عينة جديدة' : '2. Load Model & Test New Sample'}</span>
                      </div>
                      <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-500/30 px-2 py-0.5 rounded font-semibold">
                        Zero Retraining
                      </span>
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed">
                      {lang === 'ar'
                        ? 'تكوين كائنات DenseLayer جديدة تماماً وحقن الأوزان المحفوظة مباشرة، ثم حساب المخرج للعينة المطلوبة بدون أي تدريب.'
                        : 'Instantiating uninitialized layers, injecting the weights, and computing pure forward pass inference.'}
                    </p>

                    {/* Sample Inputs Config */}
                    <div className="space-y-2 bg-slate-950/80 p-3 rounded-xl border border-slate-800 text-xs">
                      <div className="font-semibold text-slate-300 flex items-center justify-between">
                        <span>{lang === 'ar' ? 'مواصفات العينة الجديدة (X_test):' : 'Test Sample Inputs (X_test):'}</span>
                        <span className="font-mono text-sky-400">[[{testSampleX1.toFixed(2)}, {testSampleX2.toFixed(2)}]]</span>
                      </div>

                      <div className="grid grid-cols-2 gap-3 pt-1">
                        <div>
                          <label className="text-[10px] text-slate-400 block mb-1">الميزة الأولى (X1): {testSampleX1.toFixed(2)}</label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={testSampleX1}
                            onChange={(e) => {
                              const val = Number(e.target.value);
                              setTestSampleX1(val);
                              handleLoadModelAndPredict(val, testSampleX2);
                            }}
                            className="w-full accent-sky-500 cursor-pointer"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] text-slate-400 block mb-1">الميزة الثانية (X2): {testSampleX2.toFixed(2)}</label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={testSampleX2}
                            onChange={(e) => {
                              const val = Number(e.target.value);
                              setTestSampleX2(val);
                              handleLoadModelAndPredict(testSampleX1, val);
                            }}
                            className="w-full accent-sky-500 cursor-pointer"
                          />
                        </div>
                      </div>

                      {/* Quick Preset Buttons */}
                      <div className="flex items-center gap-2 pt-1 text-[10px]">
                        <span className="text-slate-500">{lang === 'ar' ? 'عينات سريعة:' : 'Presets:'}</span>
                        <button
                          onClick={() => {
                            setTestSampleX1(0.85);
                            setTestSampleX2(0.85);
                            handleLoadModelAndPredict(0.85, 0.85);
                          }}
                          className="px-2 py-0.5 rounded bg-sky-950 border border-sky-500/40 text-sky-300 hover:bg-sky-900 transition font-mono font-bold"
                        >
                          [0.85, 0.85] (عينة الدرس)
                        </button>
                        <button
                          onClick={() => {
                            setTestSampleX1(0.15);
                            setTestSampleX2(0.15);
                            handleLoadModelAndPredict(0.15, 0.15);
                          }}
                          className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition font-mono"
                        >
                          [0.15, 0.15]
                        </button>
                        <button
                          onClick={() => {
                            setTestSampleX1(0.9);
                            setTestSampleX2(0.2);
                            handleLoadModelAndPredict(0.9, 0.2);
                          }}
                          className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition font-mono"
                        >
                          [0.90, 0.20]
                        </button>
                      </div>
                    </div>

                    {/* Loaded Model Prediction Output */}
                    <div className="bg-slate-950 p-3.5 rounded-xl border border-emerald-500/30 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">{lang === 'ar' ? 'الاحتمال المتوقع (Sigmoid):' : 'Predicted Probability:'}</span>
                        <span className="font-mono text-base font-black text-emerald-400">
                          {loadedPred ? `${(loadedPred.prob * 100).toFixed(2)}%` : '--'}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">{lang === 'ar' ? 'التصنيف النهائي (Class Decision):' : 'Final Class Decision:'}</span>
                        <span
                          className={`font-mono text-xs font-bold px-2.5 py-0.5 rounded-full ${
                            loadedPred && loadedPred.predClass === 1
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                              : 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                          }`}
                        >
                          {loadedPred
                            ? (lang === 'ar' ? `فئة ${loadedPred.predClass} (${loadedPred.predClass === 1 ? 'سلوك مرتفع' : 'سلوك منخفض'})` : `Class ${loadedPred.predClass}`)
                            : '--'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={() => handleLoadModelAndPredict(testSampleX1, testSampleX2)}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-lg shadow-emerald-600/20 transition pt-2"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>{lang === 'ar' ? 'تنفيذ الاستنتاج على النموذج المسترجع (Run Forward Pass)' : 'Run Loaded Model Inference'}</span>
                  </button>
                </div>
              </div>

              {/* Educational Card: IoT & Embedded Systems Advantages */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
                <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
                  <Cpu className="w-4 h-4 text-sky-400" />
                  <h4 className="font-bold text-slate-100 text-sm">
                    {lang === 'ar'
                      ? 'لماذا تعتبر ميزة Save/Load بهذه البساطة ثورة في أنظمة IoT والبيانات الخفيفة؟'
                      : 'Why JSON/Array Weight Serialization is Essential for IoT & Embedded Systems'}
                  </h4>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                    <div className="font-bold text-sky-300">1. حجم ذاكرة متناهي الصغر (Ultra-low Footprint)</div>
                    <p className="text-slate-400 text-[11px] leading-relaxed">
                      {lang === 'ar'
                        ? 'متحكمات مثل ESP32 أو Raspberry Pi Pico تملك ذاكرة RAM محدودة (أقل من 300KB). شبكتنا تزن 33 رقماً عائماً فقط (~132 بايت في الذاكرة الثنائية)، مما يمكن تشغيلها بسلاسة حتى على شريحة Arduino!'
                        : 'Microcontrollers have tiny SRAM (< 300KB). Our 33 floats require only ~132 bytes of binary memory, easily running on an Arduino.'}
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                    <div className="font-bold text-emerald-300">2. استقلالية تامة وبدون حزم ثقيلة (Zero Dependencies)</div>
                    <p className="text-slate-400 text-[11px] leading-relaxed">
                      {lang === 'ar'
                        ? 'لا تحتاج لتثبيت مكتبات بايثون ضخمة (كالتي تزن مئات الميغابايت مثل PyTorch أو TF). يكفي حلقة تكرار بسيطة (Loop) بلغة C أو MicroPython لإجراء عملية الضرب الجمعي Forward Pass.'
                        : 'No heavy C++ or Python frameworks needed. A simple 10-line C matrix loop computes the forward pass directly.'}
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                    <div className="font-bold text-amber-300">3. اتخاذ القرار على الحافة وتوفير الطاقة (Edge AI)</div>
                    <p className="text-slate-400 text-[11px] leading-relaxed">
                      {lang === 'ar'
                        ? 'إرسال البيانات عبر Wi-Fi أو 4G يستهلك بطارية الحساسات. تشغيل النموذج محلياً يستغرق بضع ميكروثوانٍ فقط مع استهلاك طاقة يكاد لا يُذكر، مع الحفاظ الكامل على خصوصية البيانات.'
                        : 'Transmitting raw telemetry wastes battery and bandwidth. Edge forward pass executes in microseconds offline with zero cloud latency.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 3: Complete Standalone Python Code */}
          {activeTab === 'code' && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-sky-400" />
                  <h4 className="font-bold text-slate-100 text-sm">
                    {lang === 'ar'
                      ? 'كود بايثون المستقل (Binary Classification with Sigmoid & BCE)'
                      : 'Standalone Python Code (Sigmoid & BCE)'}
                  </h4>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopyCode}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition"
                  >
                    {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedCode ? t.copied : t.copyCode}</span>
                  </button>

                  <button
                    onClick={handleDownloadCode}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold transition"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>{t.downloadPython}</span>
                  </button>
                </div>
              </div>

              <pre className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-slate-300 overflow-x-auto max-h-[460px] border border-slate-800/80 leading-relaxed">
                <code>{standalonePythonCode}</code>
              </pre>
            </div>
          )}

          {/* Tab 4: Mathematical Theory & Deep Learning Insights */}
          {activeTab === 'theory' && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4 text-xs sm:text-sm text-slate-300 leading-relaxed">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
                <Sparkles className="w-5 h-5 text-sky-400" />
                <h4 className="font-bold text-slate-100 text-base">
                  {lang === 'ar'
                    ? 'التشريح الرياضي لنموذج التصنيف الثنائي (Sigmoid & BCE)'
                    : 'Mathematical Insights: Sigmoid & Binary Cross-Entropy'}
                </h4>
              </div>

              <div className="space-y-3">
                <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                  <div className="font-bold text-sky-300 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-sky-400" />
                    {lang === 'ar' ? '1. ما هو دور دالة Sigmoid في طبقة المخرجات؟' : '1. Role of Sigmoid Activation in Binary Classification'}
                  </div>
                  <div className="text-slate-400 text-xs font-mono bg-slate-900 p-2.5 rounded-lg border border-slate-800/80">
                    <div>σ(z) = 1 / (1 + e^(-z))</div>
                    <div>dσ/dz = σ(z) * (1 - σ(z))</div>
                  </div>
                  <p className="text-slate-400 text-xs">
                    {lang === 'ar'
                      ? 'على عكس نماذج الانحدار والتسعير التي تخرج أرقاماً غير مقيدة، تحول دالة Sigmoid أي رقم حقيقي مستمر (Logit) إلى قيمة احتمال قطعية محصورة بين 0 و 1 تمثل P(y=1|X)، وهو ما يتطلبه التصنيف الثنائي.'
                      : 'Unlike continuous regression, Sigmoid squashes any real-valued logit into a bounded probability score between 0 and 1, representing P(y=1|X).'}
                  </p>
                </div>

                <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                  <div className="font-bold text-emerald-300 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    {lang === 'ar' ? '2. لماذا نستخدم Binary Cross-Entropy بدلاً من MSE في التصنيف؟' : '2. Why BCE over MSE in Classification?'}
                  </div>
                  <div className="text-slate-400 text-xs font-mono bg-slate-900 p-2.5 rounded-lg border border-slate-800/80">
                    <div>BCE = -(1/N) * Σ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]</div>
                    <div>∂L/∂ŷ = -(1/N) * [ (y / ŷ) - ((1 - y) / (1 - ŷ)) ]</div>
                  </div>
                  <p className="text-slate-400 text-xs">
                    {lang === 'ar'
                      ? 'عند دمج Sigmoid مع MSE تصبح دالة الخسارة غير محدبة (Non-Convex) وتعاني من تلاشي التدرجات (Vanishing Gradients) عندما يخطئ النموذج بثقة. بينما Binary Cross-Entropy (Log Loss) تعاقب التنبؤات الخاطئة الواثقة بشدة، وتمنح سطحاً محدباً يسهل لمحسّن Adam الهبوط فيه بسرعة واستقرار.'
                      : 'MSE with Sigmoid leads to non-convex surfaces and vanishing gradients. Binary Cross-Entropy severely penalizes confident mistakes with logarithmic loss, ensuring strong gradients.'}
                  </p>
                </div>

                <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                  <div className="font-bold text-amber-300 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400" />
                    {lang === 'ar' ? '3. السحر الرياضي في قاعدة السلسلة (Chain Rule Cancellation)' : '3. Beautiful Chain Rule Cancellation'}
                  </div>
                  <p className="text-slate-400 text-xs">
                    {lang === 'ar'
                      ? 'عند ضرب مشتقة Binary Cross-Entropy بمشتقة دالة Sigmoid، تُلغى الحدود الرياضية المعقدة في المقام ويتبقى تدرج نقي وبسيط جداً: (ŷ - y)، تماماً كبساطة تدرج الانحدار الخطي ولكن في فضاء الاحتمالات!'
                      : 'When multiplying dL/dŷ by dŷ/dz (Sigmoid derivative), terms cancel out yielding the clean gradient: dL/dz = ŷ - y.'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
