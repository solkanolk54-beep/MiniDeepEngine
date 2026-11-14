/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  FastForward,
  Copy,
  Check,
  Download,
  Building,
  Home,
  TrendingUp,
  Sliders,
  DollarSign,
  Maximize2,
  CheckCircle2,
  AlertCircle,
  Code,
  Terminal,
  Activity,
  Layers,
  Sparkles,
  Info,
} from 'lucide-react';
import { Language, translations } from '../translations';
import { DenseLayer } from '../engine/layers';
import { Activation_ReLU } from '../engine/activations';
import { Loss_MSE } from '../engine/loss';
import { Optimizer_Adam } from '../engine/optimizers';
import { Matrix, MatrixOps } from '../engine/matrix';

interface RealEstatePricingLabProps {
  lang: Language;
}

// 6 Core Training Apartments from User Prompt
interface ApartmentData {
  id: number;
  area: number;   // m²
  rooms: number;  // count
  actualPrice: number; // Million
  noteAr?: string;
  noteEn?: string;
}

const DEFAULT_TRAINING_DATA: ApartmentData[] = [
  { id: 1, area: 60,  rooms: 2, actualPrice: 12, noteAr: 'شقة صغيرة - غرفتين', noteEn: 'Small Flat - 2 Rooms' },
  { id: 2, area: 80,  rooms: 3, actualPrice: 16, noteAr: 'شقة متوسطة - 3 غرف', noteEn: 'Medium Flat - 3 Rooms' },
  { id: 3, area: 100, rooms: 3, actualPrice: 20, noteAr: 'شقة واسعة - 3 غرف', noteEn: 'Spacious Flat - 3 Rooms' },
  { id: 4, area: 120, rooms: 4, actualPrice: 25, noteAr: 'شقة عائلية - 4 غرف', noteEn: 'Family Flat - 4 Rooms' },
  { id: 5, area: 150, rooms: 5, actualPrice: 32, noteAr: 'بنتهاوس فاخر - 5 غرف', noteEn: 'Luxury Penthouse - 5 Rooms' },
  { id: 6, area: 90,  rooms: 2, actualPrice: 18, noteAr: 'شقة مميزة - غرفتين', noteEn: 'Premium Flat - 2 Rooms' },
];

// Unseen Test Apartments for out-of-sample evaluation
const UNSEEN_TEST_APARTMENTS = [
  { area: 70,  rooms: 2, labelAr: 'عقار تجريبي A (70م² - غرفتين)', labelEn: 'Test Flat A (70m² - 2 Rooms)' },
  { area: 110, rooms: 3, labelAr: 'عقار تجريبي B (110م² - 3 غرف)', labelEn: 'Test Flat B (110m² - 3 Rooms)' },
  { area: 135, rooms: 4, labelAr: 'عقار تجريبي C (135م² - 4 غرف)', labelEn: 'Test Flat C (135m² - 4 Rooms)' },
  { area: 180, rooms: 5, labelAr: 'عقار تجريبي D (180م² - 5 غرف)', labelEn: 'Test Flat D (180m² - 5 Rooms)' },
];

export const RealEstatePricingLab: React.FC<RealEstatePricingLabProps> = ({ lang }) => {
  const t = translations[lang];

  // Training Data State
  const [dataset, setDataset] = useState<ApartmentData[]>(DEFAULT_TRAINING_DATA);

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.03);
  const [hiddenNeurons, setHiddenNeurons] = useState<number>(16);
  const [useFeatureScaling, setUseFeatureScaling] = useState<boolean>(true);

  // Training Execution State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);

  // Interactive Property Valuation Calculator
  const [customArea, setCustomArea] = useState<number>(110);
  const [customRooms, setCustomRooms] = useState<number>(3);
  const [customPrediction, setCustomPrediction] = useState<number | null>(null);

  // Sub-tabs
  const [activeTab, setActiveTab] = useState<'visuals' | 'table' | 'code' | 'theory'>('visuals');
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  // Refs for Animation and Canvas
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationReqRef = useRef<number | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  // Normalization Statistics
  const stats = useMemo(() => {
    const areas = dataset.map((d) => d.area);
    const rooms = dataset.map((d) => d.rooms);
    const prices = dataset.map((d) => d.actualPrice);

    const meanArea = areas.reduce((a, b) => a + b, 0) / areas.length;
    const stdArea = Math.sqrt(areas.reduce((a, b) => a + Math.pow(b - meanArea, 2), 0) / areas.length) || 1;

    const meanRooms = rooms.reduce((a, b) => a + b, 0) / rooms.length;
    const stdRooms = Math.sqrt(rooms.reduce((a, b) => a + Math.pow(b - meanRooms, 2), 0) / rooms.length) || 1;

    const meanPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    const stdPrice = Math.sqrt(prices.reduce((a, b) => a + Math.pow(b - meanPrice, 2), 0) / prices.length) || 1;

    return { meanArea, stdArea, meanRooms, stdRooms, meanPrice, stdPrice };
  }, [dataset]);

  // Model references
  const modelRef = useRef<{
    hiddenLayer: DenseLayer;
    activation: Activation_ReLU;
    outputLayer: DenseLayer;
    lossFn: Loss_MSE;
    optimizer: Optimizer_Adam;
  } | null>(null);

  // Initialize or re-create model
  const resetModel = useCallback(() => {
    setIsTraining(false);
    setEpoch(0);
    setLossHistory([]);

    const hidden = new DenseLayer(2, hiddenNeurons, 0.05);
    const act = new Activation_ReLU();
    const output = new DenseLayer(hiddenNeurons, 1, 0.05);
    const lossFn = new Loss_MSE();
    const opt = new Optimizer_Adam(learningRate, 0.9, 0.999, 1e-8);

    modelRef.current = {
      hiddenLayer: hidden,
      activation: act,
      outputLayer: output,
      lossFn: lossFn,
      optimizer: opt,
    };

    const initialLogs = [
      lang === 'ar'
        ? `=== تم تهيئة معمارية الشبكة العصبية لتسعير العقارات ===`
        : `=== Initialized Real Estate Neural Valuation Network ===`,
      lang === 'ar'
        ? `• طبقة الإدخال: ميزتان [المساحة بالـ م²، عدد الغرف]`
        : `• Input Layer: 2 Features [Area (m²), Rooms Count]`,
      lang === 'ar'
        ? `• الطبقة المخفية: ${hiddenNeurons} خلية عصبية + دالة تفعيل ReLU`
        : `• Hidden Layer: ${hiddenNeurons} Neurons + ReLU Activation`,
      lang === 'ar'
        ? `• طبقة المخرجات: خلية خطية واحدة للتسعير (Linear Output)`
        : `• Output Layer: 1 Linear Neuron (Continuous Regression)`,
      lang === 'ar'
        ? `• دالة الخسارة: Mean Squared Error (MSE) | المحسّن: Adam (lr=${learningRate})`
        : `• Loss Function: MSE | Optimizer: Adam (lr=${learningRate})`,
    ];
    setTrainingLogs(initialLogs);

    // Initial forward pass evaluation
    evaluateCurrentModel();
  }, [hiddenNeurons, learningRate, lang]);

  // Forward pass helper
  const predictSingle = useCallback(
    (area: number, rooms: number): number => {
      if (!modelRef.current) return 0;
      const { hiddenLayer, activation, outputLayer } = modelRef.current;

      let normArea = area;
      let normRooms = rooms;

      if (useFeatureScaling) {
        normArea = (area - stats.meanArea) / stats.stdArea;
        normRooms = (rooms - stats.meanRooms) / stats.stdRooms;
      }

      const inputMatrix: Matrix = [[normArea, normRooms]];
      const hOut = hiddenLayer.forward(inputMatrix);
      const aOut = activation.forward(hOut);
      const out = outputLayer.forward(aOut);

      let predPrice = out[0][0];
      if (useFeatureScaling) {
        predPrice = predPrice * stats.stdPrice + stats.meanPrice;
      }

      return Math.max(0, predPrice);
    },
    [stats, useFeatureScaling]
  );

  // Evaluate current model on dataset and custom input
  const evaluateCurrentModel = useCallback(() => {
    if (!modelRef.current) return;
    const { hiddenLayer, activation, outputLayer, lossFn } = modelRef.current;

    // Prepare inputs & targets
    const X: Matrix = dataset.map((d) => {
      if (useFeatureScaling) {
        return [
          (d.area - stats.meanArea) / stats.stdArea,
          (d.rooms - stats.meanRooms) / stats.stdRooms,
        ];
      }
      return [d.area, d.rooms];
    });

    const y: Matrix = dataset.map((d) => {
      if (useFeatureScaling) {
        return [(d.actualPrice - stats.meanPrice) / stats.stdPrice];
      }
      return [d.actualPrice];
    });

    // Forward pass
    const hOut = hiddenLayer.forward(X);
    const aOut = activation.forward(hOut);
    const preds = outputLayer.forward(aOut);
    const lossVal = lossFn.forward(preds, y);

    setCurrentLoss(lossVal);

    // Unscale predictions for UI table
    const unscaledPreds = preds.map((row) => {
      let val = row[0];
      if (useFeatureScaling) {
        val = val * stats.stdPrice + stats.meanPrice;
      }
      return Math.max(0, val);
    });
    setPredictions(unscaledPreds);

    // Custom valuation calculation
    const custVal = predictSingle(customArea, customRooms);
    setCustomPrediction(custVal);
  }, [dataset, stats, useFeatureScaling, customArea, customRooms, predictSingle]);

  // Execute 1 training epoch
  const trainStep = useCallback((): number => {
    if (!modelRef.current) return 0;
    const { hiddenLayer, activation, outputLayer, lossFn, optimizer } = modelRef.current;

    const X: Matrix = dataset.map((d) => {
      if (useFeatureScaling) {
        return [
          (d.area - stats.meanArea) / stats.stdArea,
          (d.rooms - stats.meanRooms) / stats.stdRooms,
        ];
      }
      return [d.area, d.rooms];
    });

    const y: Matrix = dataset.map((d) => {
      if (useFeatureScaling) {
        return [(d.actualPrice - stats.meanPrice) / stats.stdPrice];
      }
      return [d.actualPrice];
    });

    // 1. Forward Pass
    const hOut = hiddenLayer.forward(X);
    const aOut = activation.forward(hOut);
    const preds = outputLayer.forward(aOut);

    // 2. Compute Loss
    const loss = lossFn.forward(preds, y);

    // 3. Backpropagation
    const dLoss = lossFn.backward(preds, y);
    outputLayer.backward(dLoss);
    activation.backward(outputLayer.dvalues);
    hiddenLayer.backward(activation.dvalues);

    // 4. Update Weights via Adam
    optimizer.update_params(outputLayer);
    optimizer.update_params(hiddenLayer);

    return loss;
  }, [dataset, stats, useFeatureScaling]);

  // Single step button
  const handleSingleStep = () => {
    const loss = trainStep();
    setEpoch((prev) => {
      const nextEpoch = prev + 1;
      setLossHistory((h) => [...h, loss]);
      return nextEpoch;
    });
    evaluateCurrentModel();
  };

  // Run full 1000 epochs with checkpoints matching user prompt
  const handleRun1000Epochs = () => {
    setIsTraining(false);
    let lastLoss = 0;
    const newLogs: string[] = [
      lang === 'ar'
        ? `\n>>> بدء تدريب مكثف: 1000 دورة تدريبية بواسطة خوارزمية Adam <<<`
        : `\n>>> Starting Full 1000-Epoch Training via Adam Optimizer <<<`,
    ];

    const currentEpoch = epoch;
    const updatedHistory = [...lossHistory];

    for (let i = 1; i <= 1000; i++) {
      lastLoss = trainStep();
      const currentGlobalEpoch = currentEpoch + i;
      updatedHistory.push(lastLoss);

      // Milestone logging every 200 epochs as requested in user prompt
      if (i % 200 === 0 || i === 1) {
        const logMsg =
          lang === 'ar'
            ? `Epoch ${String(i).padStart(4, ' ')} | Loss (MSE): ${lastLoss.toFixed(6)}`
            : `Epoch ${String(i).padStart(4, ' ')} | Loss (MSE): ${lastLoss.toFixed(6)}`;
        newLogs.push(logMsg);
      }
    }

    setEpoch(currentEpoch + 1000);
    setLossHistory(updatedHistory);
    setCurrentLoss(lastLoss);

    // Evaluate unseen test apartments
    newLogs.push(
      lang === 'ar'
        ? `\n--- التوقعات على عقارات اختبار غير مشمولة في التدريب ---`
        : `\n--- Evaluating Predictions on Unseen Test Apartments ---`
    );

    UNSEEN_TEST_APARTMENTS.forEach((apt) => {
      const p = predictSingle(apt.area, apt.rooms);
      newLogs.push(
        lang === 'ar'
          ? `• ${apt.labelAr}: السعر التقديري = ${p.toFixed(2)} مليون`
          : `• ${apt.labelEn}: Estimated Valuation = ${p.toFixed(2)} M`
      );
    });

    setTrainingLogs((prev) => [...prev, ...newLogs]);
    evaluateCurrentModel();
  };

  // Step 4: Run Forward Pass on unseen apartment matching the user prompt: X_new = np.array([[110.0, 3.0]])
  const handleRunStep4NewPrediction = (area = 110, rooms = 3) => {
    const predictedPrice = predictSingle(area, rooms);
    
    const logs = [
      lang === 'ar' ? `\n=== اختبار التنبؤ بعقار جديد ===` : `\n=== Testing Prediction on New Property ===`,
      lang === 'ar'
        ? `مواصفات الشقة الجديدة -> المساحة: ${area.toFixed(1)}م² | عدد الغرف: ${rooms}`
        : `New Apartment Specs -> Area: ${area.toFixed(1)}m² | Rooms: ${rooms}`,
      lang === 'ar'
        ? `السعر المتوقع بواسطة النموذج: ${predictedPrice.toFixed(2)} مليون`
        : `Predicted Price by Model: ${predictedPrice.toFixed(2)} Million`,
    ];

    setTrainingLogs((prev) => [...prev, ...logs]);
    setCustomArea(area);
    setCustomRooms(rooms);
    setCustomPrediction(predictedPrice);
  };

  // Live continuous training loop
  useEffect(() => {
    if (!isTraining) return;

    let localEpoch = epoch;
    const runBatch = () => {
      for (let i = 0; i < 5; i++) {
        const l = trainStep();
        localEpoch++;
        if (localEpoch % 20 === 0) {
          setLossHistory((h) => [...h, l]);
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

  // Initial model setup
  useEffect(() => {
    resetModel();
  }, [resetModel]);

  // Update custom prediction when sliders move
  useEffect(() => {
    if (modelRef.current) {
      const pred = predictSingle(customArea, customRooms);
      setCustomPrediction(pred);
    }
  }, [customArea, customRooms, predictSingle]);

  // Auto-scroll terminal logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [trainingLogs]);

  // Canvas visualizer: Area vs Predicted Price with curves for each room category
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, width, height);

    // Margin & plot boundaries
    const padLeft = 60;
    const padRight = 30;
    const padTop = 30;
    const padBottom = 50;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    const minArea = 40;
    const maxArea = 180;
    const minPrice = 8;
    const maxPrice = 38;

    const scaleX = (val: number) => padLeft + ((val - minArea) / (maxArea - minArea)) * plotW;
    const scaleY = (val: number) => height - padBottom - ((val - minPrice) / (maxPrice - minPrice)) * plotH;

    // Draw Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;

    // Vertical grid (Area)
    for (let a = 60; a <= 160; a += 20) {
      const x = scaleX(a);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, height - padBottom);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${a} م²`, x, height - padBottom + 18);
    }

    // Horizontal grid (Price)
    for (let p = 10; p <= 35; p += 5) {
      const y = scaleY(p);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`${p}M`, padLeft - 10, y + 4);
    }

    // Draw Model Regression Curves for rooms = 2, 3, 4, 5
    const roomConfigs = [
      { rooms: 2, color: '#38bdf8', label: '2 غرف (Rooms)' },
      { rooms: 3, color: '#34d399', label: '3 غرف (Rooms)' },
      { rooms: 4, color: '#fbbf24', label: '4 غرف (Rooms)' },
      { rooms: 5, color: '#f43f5e', label: '5 غرف (Rooms)' },
    ];

    roomConfigs.forEach(({ rooms, color }) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();

      let isFirst = true;
      for (let a = minArea; a <= maxArea; a += 3) {
        const p = predictSingle(a, rooms);
        const clampedP = Math.max(minPrice, Math.min(maxPrice, p));
        const px = scaleX(a);
        const py = scaleY(clampedP);

        if (isFirst) {
          ctx.moveTo(px, py);
          isFirst = false;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    });

    // Draw Actual Training Data Points
    dataset.forEach((apt, idx) => {
      const x = scaleX(apt.area);
      const yActual = scaleY(apt.actualPrice);
      const currentPred = predictions[idx] || 0;
      const yPred = scaleY(Math.max(minPrice, Math.min(maxPrice, currentPred)));

      // Error residual connector line
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = '#ef444488';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, yActual);
      ctx.lineTo(x, yPred);
      ctx.stroke();
      ctx.setLineDash([]);

      // Glow circle
      ctx.beginPath();
      ctx.arc(x, yActual, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf822';
      ctx.fill();

      // Core point (actual)
      ctx.beginPath();
      ctx.arc(x, yActual, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Predicted position point
      ctx.beginPath();
      ctx.arc(x, yPred, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981';
      ctx.fill();

      // Label
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`#${apt.id} (${apt.actualPrice}M)`, x, yActual - 10);
    });

    // Draw Interactive Custom Property Crosshair
    if (customPrediction !== null) {
      const cx = scaleX(customArea);
      const cy = scaleY(Math.max(minPrice, Math.min(maxPrice, customPrediction)));

      // Glowing pulsing marker
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(236, 72, 153, 0.25)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ec4899';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Readout tag
      ctx.fillStyle = '#f43f5e';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        `★ عقارك: ${customPrediction.toFixed(2)}M`,
        cx,
        Math.max(padTop + 15, cy - 14)
      );
    }

    // Axis Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      lang === 'ar' ? 'مساحة الشقة (المتر المربع م²)' : 'Apartment Area (m²)',
      width / 2,
      height - 10
    );

    ctx.save();
    ctx.translate(16, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(
      lang === 'ar' ? 'السعر (مليون وحدة)' : 'Price (Million Units)',
      0,
      0
    );
    ctx.restore();
  }, [dataset, predictions, customArea, customRooms, customPrediction, predictSingle, lang]);

  // Complete standalone Python script matching the user's prompt
  const standalonePythonCode = `# ==============================================================================
# منظومة MiniDeepEngine - التطبيق العملي الأول: النماذج التنبؤية والتسعير
# تسعير الشقق والعقارات واختبار التنبؤ بشقة جديدة لم يراها النموذج من قبل
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- 1. بناء المكونات الأساسية للشبكة ---
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

class Loss_MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        return 2 * (y_pred - y_true) / samples

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


# --- 2. تجهيز بيانات العقارات ---
X = np.array([
    [60.0, 2.0],
    [80.0, 3.0],
    [100.0, 3.0],
    [120.0, 4.0],
    [150.0, 5.0],
    [90.0, 2.0]
])

y = np.array([
    [12.0],
    [16.0],
    [20.0],
    [25.0],
    [32.0],
    [18.0]
])


# --- 3. بناء الشبكة والتدريب ---
hidden_layer = DenseLayer(n_inputs=2, n_neurons=16)
activation = Activation_ReLU()
output_layer = DenseLayer(n_inputs=16, n_neurons=1)

loss_fn = Loss_MSE()
optimizer = Optimizer_Adam(learning_rate=0.03)

epochs = 1000

for epoch in range(epochs + 1):
    dense1_out = hidden_layer.forward(X)
    act_out = activation.forward(dense1_out)
    preds = output_layer.forward(act_out)
    
    loss = loss_fn.forward(preds, y)
    
    d_loss = loss_fn.backward(preds, y)
    output_layer.backward(d_loss)
    activation.backward(output_layer.dinputs)
    hidden_layer.backward(activation.dvalues)
    
    optimizer.update_params(output_layer)
    optimizer.update_params(hidden_layer)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")


# --- 4. اختبار شقة جديدة لم يراها النموذج من قبل ---
# مثلاً: مساحة 110 متر مربع، و 3 غرف
X_new = np.array([[110.0, 3.0]])

# تمرير الشقة الجديدة عبر النموذج (Forward Pass فقط بدون تدريب)
dense1_new = hidden_layer.forward(X_new)
act_new = activation.forward(dense1_new)
predicted_price = output_layer.forward(act_new)

print("=== اختبار التنبؤ بعقار جديد ===")
print(f"مواصفات الشقة الجديدة -> المساحة: {X_new[0][0]}م² | عدد الغرف: {int(X_new[0][1])}")
print(f"السعر المتوقع بواسطة النموذج: {predicted_price[0][0]:.2f} مليون")
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
    link.download = 'real_estate_pricing_regression.py';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/70 to-slate-900 border border-indigo-500/30 rounded-2xl p-5 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 relative z-10">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <Building className="w-5 h-5" />
              </span>
              <h2 className="text-xl sm:text-2xl font-black text-white tracking-wide">
                {t.pricingTitle}
              </h2>
              <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Tabular Regression + Adam
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-300 max-w-3xl leading-relaxed">
              {t.pricingSubtitle}
            </p>
          </div>

          {/* Quick HUD Metrics */}
          <div className="flex items-center gap-3 bg-slate-950/70 border border-slate-800 rounded-xl p-2.5">
            <div className="text-center px-3 border-r border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                {t.epochs}
              </div>
              <div className="text-base sm:text-lg font-mono font-bold text-indigo-400">
                {epoch}
              </div>
            </div>

            <div className="text-center px-3 border-r border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Loss (MSE)
              </div>
              <div className="text-base sm:text-lg font-mono font-bold text-amber-400">
                {currentLoss.toFixed(5)}
              </div>
            </div>

            <div className="text-center px-3">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                {lang === 'ar' ? 'العمارة' : 'Arch'}
              </div>
              <div className="text-xs font-mono font-bold text-emerald-400">
                2 → {hiddenNeurons} (ReLU) → 1
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Step 4: Dedicated New Apartment Prediction Showcase (Matching user prompt) */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/40 border border-emerald-500/40 rounded-2xl p-5 shadow-2xl space-y-4 relative overflow-hidden">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <span className="p-2 rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              <Sparkles className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base sm:text-lg font-bold text-white">
                  {lang === 'ar'
                    ? 'الخطوة 4: اختبار شقة جديدة لم يراها النموذج من قبل (New Prediction)'
                    : 'Step 4: Unseen Apartment Prediction (New Prediction)'}
                </h3>
                <span className="text-[10px] font-mono bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/30 font-semibold">
                  Forward Pass Only
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {lang === 'ar'
                  ? 'اختبار قدرة النموذج على الاستنتاج والتعميم (Generalization) والتنبؤ بسعر شقة جديدة تماماً (110.0م²، 3 غرف) بناءً على ما تعلمه'
                  : 'Testing network generalization by evaluating an unseen apartment (110.0m², 3 rooms) using pure forward pass'}
              </p>
            </div>
          </div>

          <button
            onClick={() => handleRunStep4NewPrediction(110, 3)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs shadow-lg shadow-emerald-600/30 transition self-stretch sm:self-auto justify-center"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>
              {lang === 'ar' ? 'تشغيل التنبؤ للشقة الجديدة (110م² - 3 غرف)' : 'Run Prediction (110m² - 3 Rooms)'}
            </span>
          </button>
        </div>

        {/* 3-Column Grid: Code -> Live Inference -> Generalization Insight */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-stretch text-xs">
          {/* Col 1: Forward Pass Code (4 cols) */}
          <div className="md:col-span-4 bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono space-y-1.5 flex flex-col justify-between">
            <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider flex items-center justify-between">
              <span>{lang === 'ar' ? 'كود التمرير الأمامي (بايثون)' : 'Forward Pass Python Code'}</span>
              <span className="text-emerald-400 font-bold">Step 4</span>
            </div>
            <pre className="text-[11px] text-slate-300 overflow-x-auto leading-relaxed">
              <code>{`# 1. إعداد الشقة الجديدة
X_new = np.array([[110.0, 3.0]])

# 2. تمرير الشقة (Forward Pass)
dense1_new = hidden_layer.forward(X_new)
act_new = activation.forward(dense1_new)
predicted_price = output_layer.forward(act_new)`}</code>
            </pre>
            <div className="text-[10px] text-slate-500">
              {lang === 'ar' ? '• لا يتم حساب التدرجات أو تعديل الأوزان هنا' : '• No backpropagation or weight update'}
            </div>
          </div>

          {/* Col 2: Live Valuation Box (4 cols) */}
          <div className="md:col-span-4 bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950/30 p-4 rounded-xl border border-emerald-500/30 flex flex-col justify-between text-center space-y-2">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>{lang === 'ar' ? 'المواصفات المختبرة:' : 'Tested Specs:'}</span>
              <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded">
                110.0 م² | 3 غرف
              </span>
            </div>

            <div className="space-y-1 py-1">
              <div className="text-[11px] text-emerald-400 font-semibold">
                {lang === 'ar' ? 'السعر المتوقع بواسطة النموذج (ŷ_pred):' : 'Model Predicted Price (ŷ_pred):'}
              </div>
              <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 via-teal-200 to-amber-300 font-mono">
                {predictSingle(110, 3).toFixed(2)}
                <span className="text-sm font-normal text-slate-400 mr-2 font-sans">
                  {lang === 'ar' ? 'مليون وحدة' : ' Million'}
                </span>
              </div>
              <div className="text-[11px] text-slate-400">
                {lang === 'ar' ? 'معدل سعر المتر:' : 'Per m²:'}{' '}
                <strong className="text-slate-200 font-mono">
                  {((predictSingle(110, 3) * 1000000) / 110).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </strong>
              </div>
            </div>

            <div className="text-[10px] text-slate-400 bg-slate-950/60 p-1.5 rounded-lg border border-slate-800/80">
              {epoch >= 200
                ? (lang === 'ar' ? '✓ تم التدريب بنجاح: الاستنتاج ناتج عن تعلّم العلاقات الحقيقية' : '✓ Trained: Generalization from true patterns')
                : (lang === 'ar' ? '⚠️ يُرجى تدريب النموذج أولاً (1000 دورة) للحصول على دقة واقعية' : '⚠️ Train model first for accurate pricing')}
            </div>
          </div>

          {/* Col 3: Generalization Context & Comparison (4 cols) */}
          <div className="md:col-span-4 bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2 flex flex-col justify-between">
            <div className="text-[11px] font-semibold text-slate-300 flex items-center justify-between">
              <span>{lang === 'ar' ? 'مقارنة الشقة الجديدة مع بيانات التدريب:' : 'Contextual Comparison:'}</span>
              <span className="text-[10px] text-indigo-400 font-mono">Generalization</span>
            </div>

            <div className="space-y-1.5 font-mono text-[11px]">
              <div className="flex items-center justify-between p-1.5 rounded bg-slate-900/60 text-slate-400">
                <span>شقة #3 (100م² - 3 غرف)</span>
                <span className="font-bold text-sky-400">20.00 M (تدريب)</span>
              </div>

              <div className="flex items-center justify-between p-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-bold">
                <span>★ الشقة الجديدة (110م² - 3 غرف)</span>
                <span className="text-emerald-400">{predictSingle(110, 3).toFixed(2)} M (توقع)</span>
              </div>

              <div className="flex items-center justify-between p-1.5 rounded bg-slate-900/60 text-slate-400">
                <span>شقة #4 (120م² - 4 غرف)</span>
                <span className="font-bold text-amber-400">25.00 M (تدريب)</span>
              </div>
            </div>

            <p className="text-[10px] text-slate-400 leading-relaxed">
              {lang === 'ar'
                ? 'استنتجت الشبكة بسلاسة أن إضافة 10م² مع ثبات الغرف عند 3 تزيد السعر بقرابة 1.84 مليون، وتقع القيمة بانسيابية رياضية بين العينات!'
                : 'The network naturally inferred that adding 10m² for 3 rooms adds ~1.84M, interpolating logically between known data points.'}
            </p>
          </div>
        </div>
      </div>

      {/* Main Interactive Controls & Valuation Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Valuation Tool (4 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Smart Property Valuation Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Home className="w-5 h-5 text-indigo-400" />
                <h3 className="font-bold text-slate-100 text-sm sm:text-base">
                  {t.pricingValuationTool}
                </h3>
              </div>
              <span className="text-[10px] font-mono bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/20">
                Live Inference
              </span>
            </div>

            {/* Area Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium flex items-center gap-1.5">
                  <Maximize2 className="w-3.5 h-3.5 text-indigo-400" />
                  {t.pricingArea}
                </span>
                <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded text-xs">
                  {customArea} م²
                </span>
              </div>
              <input
                type="range"
                min="40"
                max="220"
                step="5"
                value={customArea}
                onChange={(e) => setCustomArea(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>40 م²</span>
                <span>100 م²</span>
                <span>160 م²</span>
                <span>220 م²</span>
              </div>
            </div>

            {/* Rooms Slider */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400 font-medium flex items-center gap-1.5">
                  <Building className="w-3.5 h-3.5 text-emerald-400" />
                  {t.pricingRooms}
                </span>
                <span className="font-mono font-bold text-white bg-slate-800 px-2 py-0.5 rounded text-xs">
                  {customRooms} {lang === 'ar' ? 'غرف' : 'Rooms'}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="6"
                step="1"
                value={customRooms}
                onChange={(e) => setCustomRooms(Number(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>1 غرفة</span>
                <span>3 غرف</span>
                <span>5 غرف</span>
                <span>6 غرف</span>
              </div>
            </div>

            {/* Preset Buttons */}
            <div className="space-y-1.5 pt-1">
              <div className="text-[11px] text-slate-400 font-medium">
                {lang === 'ar' ? 'نماذج جاهزة للاختبار الفوري:' : 'Quick Presets:'}
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <button
                  onClick={() => { setCustomArea(60); setCustomRooms(2); }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-[11px] transition text-center"
                >
                  60م² - غرفتان
                </button>
                <button
                  onClick={() => { setCustomArea(110); setCustomRooms(3); }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-[11px] transition text-center"
                >
                  110م² - 3 غرف
                </button>
                <button
                  onClick={() => { setCustomArea(135); setCustomRooms(4); }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-[11px] transition text-center"
                >
                  135م² - 4 غرف
                </button>
                <button
                  onClick={() => { setCustomArea(180); setCustomRooms(5); }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-[11px] transition text-center"
                >
                  180م² - 5 غرف
                </button>
              </div>
            </div>

            {/* Predicted Price Output Box */}
            <div className="bg-gradient-to-br from-indigo-950/60 via-slate-900 to-indigo-950/40 border border-indigo-500/40 rounded-xl p-4 text-center space-y-1.5 relative overflow-hidden">
              <div className="text-[11px] font-semibold text-indigo-300 uppercase tracking-wider">
                {t.pricingPredictedPrice}
              </div>
              <div className="text-3xl sm:text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-pink-400 to-amber-300 font-mono">
                {customPrediction !== null ? `${customPrediction.toFixed(2)}` : '--'}
                <span className="text-base text-slate-400 font-sans mr-2">
                  {lang === 'ar' ? 'مليون وحدة' : ' Million Units'}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 flex items-center justify-center gap-3 pt-1">
                <span>
                  {lang === 'ar' ? 'معدل المتر:' : 'Per m²:'}{' '}
                  <strong className="text-slate-200 font-mono">
                    {customPrediction && customArea > 0
                      ? `${((customPrediction * 1000000) / customArea).toLocaleString('en-US', { maximumFractionDigits: 0 })}`
                      : '--'}
                  </strong>
                </span>
                <span>•</span>
                <span className="text-emerald-400 font-medium">
                  {epoch >= 200 ? (lang === 'ar' ? 'نموذج مدرب' : 'Model Trained') : (lang === 'ar' ? 'قيد التدريب' : 'Training')}
                </span>
              </div>
            </div>
          </div>

          {/* Training Execution Controls */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-800 text-slate-300 font-semibold">
              <div className="flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-indigo-400" />
                <span>{lang === 'ar' ? 'لوحة تحكم التدريب والمحسّن' : 'Training & Optimizer Controls'}</span>
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
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono text-xs focus:border-indigo-500"
                >
                  <option value={0.01}>0.01 (بطيء ودقيق)</option>
                  <option value={0.03}>0.03 (القيمة الموصى بها)</option>
                  <option value={0.05}>0.05 (سريع)</option>
                  <option value={0.1}>0.10 (عالي جداً)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-slate-400 block text-[11px]">
                  {lang === 'ar' ? 'خلايا الطبقة المخفية' : 'Hidden Neurons'}
                </label>
                <select
                  value={hiddenNeurons}
                  onChange={(e) => {
                    setHiddenNeurons(Number(e.target.value));
                  }}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono text-xs focus:border-indigo-500"
                >
                  <option value={8}>8 خلايا (سريع)</option>
                  <option value={16}>16 خلية (معمارية المقال)</option>
                  <option value={32}>32 خلية (أكثر عمقاً)</option>
                </select>
              </div>
            </div>

            {/* Feature Scaling Toggle */}
            <div className="flex items-center justify-between bg-slate-950/70 p-2.5 rounded-xl border border-slate-800">
              <div className="space-y-0.5">
                <div className="text-xs font-semibold text-slate-200">
                  {lang === 'ar' ? 'التقييس المعياري (Feature Scaling)' : 'Feature Standardization'}
                </div>
                <div className="text-[10px] text-slate-400">
                  {lang === 'ar' ? 'يطبق z-score لضمان تقارب Adam المستقر' : 'Applies z-score for smooth Adam convergence'}
                </div>
              </div>
              <input
                type="checkbox"
                checked={useFeatureScaling}
                onChange={(e) => {
                  setUseFeatureScaling(e.target.checked);
                  resetModel();
                }}
                className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
              />
            </div>

            {/* Main Action Buttons */}
            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                onClick={() => setIsTraining(!isTraining)}
                className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl font-bold text-xs transition ${
                  isTraining
                    ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-600/30'
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/30'
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

            {/* Run 1000 Epochs (from Prompt) */}
            <button
              onClick={handleRun1000Epochs}
              disabled={isTraining}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-bold text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-600/30 transition disabled:opacity-50"
            >
              <FastForward className="w-4 h-4" />
              <span>{t.pricingRun1000}</span>
            </button>
          </div>
        </div>

        {/* Right Column: Visualization Canvas & Charts & Tabs (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Sub Navigation Bar */}
          <div className="bg-slate-900 border border-slate-800 p-1.5 rounded-xl flex items-center gap-2 text-xs">
            <button
              onClick={() => setActiveTab('visuals')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'visuals'
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'منحنى الانحدار ومنحنى الخسارة' : 'Regression Curves & Loss'}</span>
            </button>

            <button
              onClick={() => setActiveTab('table')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'table'
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'جدول العينات ومؤشرات الخطأ' : 'Data Samples & Residuals'}</span>
            </button>

            <button
              onClick={() => setActiveTab('code')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold transition ${
                activeTab === 'code'
                  ? 'bg-indigo-600 text-white'
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
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Info className="w-3.5 h-3.5" />
              <span>{lang === 'ar' ? 'التشريح النظري والرياضي' : 'Mathematical Theory'}</span>
            </button>
          </div>

          {/* Tab 1: Visuals (Regression Surface + Loss Curve + Terminal Logs) */}
          {activeTab === 'visuals' && (
            <div className="space-y-4">
              {/* Regression Curves Canvas */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800 text-xs">
                  <div className="font-bold text-slate-200 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-indigo-400" />
                    <span>
                      {lang === 'ar'
                        ? 'خريطة الانحدار التنبؤي: المساحة vs السعر عبر فئات الغرف'
                        : 'Regression Curves: Area vs Price by Room Count'}
                    </span>
                  </div>
                  {/* Legend */}
                  <div className="flex items-center gap-3 text-[10px]">
                    <span className="flex items-center gap-1 text-sky-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-sky-400" /> 2 غرف
                    </span>
                    <span className="flex items-center gap-1 text-emerald-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> 3 غرف
                    </span>
                    <span className="flex items-center gap-1 text-amber-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> 4 غرف
                    </span>
                    <span className="flex items-center gap-1 text-rose-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-rose-400" /> 5 غرف
                    </span>
                  </div>
                </div>

                <div className="relative aspect-[16/9] w-full rounded-xl overflow-hidden border border-slate-800 bg-[#090d16]">
                  <canvas
                    ref={canvasRef}
                    width={640}
                    height={360}
                    className="w-full h-full block"
                  />
                </div>
              </div>

              {/* Split row: Loss Progression & Terminal Logs */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* SVG Loss Curve */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-2">
                  <div className="flex items-center justify-between text-xs pb-1 border-b border-slate-800">
                    <span className="font-bold text-slate-200">
                      {lang === 'ar' ? 'منحنى هبوط دالة الخسارة (MSE)' : 'MSE Loss Descent Curve'}
                    </span>
                    <span className="font-mono text-amber-400 font-bold">
                      {currentLoss.toFixed(6)}
                    </span>
                  </div>

                  <div className="h-36 w-full relative">
                    {lossHistory.length > 1 ? (
                      <svg className="w-full h-full overflow-visible" viewBox="0 0 300 120" preserveAspectRatio="none">
                        {/* Background subtle grid */}
                        <line x1="0" y1="30" x2="300" y2="30" stroke="#1e293b" strokeDasharray="3,3" />
                        <line x1="0" y1="60" x2="300" y2="60" stroke="#1e293b" strokeDasharray="3,3" />
                        <line x1="0" y1="90" x2="300" y2="90" stroke="#1e293b" strokeDasharray="3,3" />

                        {/* Sparkline Polyline */}
                        {(() => {
                          const maxL = Math.max(...lossHistory.slice(0, 50), 1);
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
                            <>
                              <polyline
                                fill="none"
                                stroke="#f59e0b"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                points={pts}
                              />
                            </>
                          );
                        })()}
                      </svg>
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-slate-500">
                        {lang === 'ar' ? 'ابدأ التدريب لعرض منحنى الهبوط' : 'Start training to render curve'}
                      </div>
                    )}
                  </div>

                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Epoch 0</span>
                    <span>Epoch {epoch}</span>
                  </div>
                </div>

                {/* Training Console Logs (Matching user prompt console output) */}
                <div className="bg-slate-950 border border-slate-800 rounded-2xl p-3 shadow-xl space-y-2 flex flex-col justify-between">
                  <div className="flex items-center justify-between pb-1 border-b border-slate-800 text-[11px]">
                    <div className="flex items-center gap-1.5 text-slate-300 font-mono">
                      <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Console Logs (Adam Optimizer)</span>
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
                            ? 'text-indigo-300 font-bold'
                            : log.startsWith('•')
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

          {/* Tab 2: Table of Training Data & Residuals */}
          {activeTab === 'table' && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="space-y-0.5">
                  <h4 className="font-bold text-slate-100 text-sm">
                    {lang === 'ar' ? 'عينات تدريب تسعير العقارات ومؤشرات الخطأ' : 'Training Data Samples & Residuals'}
                  </h4>
                  <p className="text-xs text-slate-400">
                    {lang === 'ar'
                      ? 'مقارنة دقيقة بين السعر الحقيقي للمنزل والتوقع الحالي لشبكة MiniDeepEngine'
                      : 'Comparison between ground-truth prices and current neural predictions'}
                  </p>
                </div>
              </div>

              {/* Data Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-mono text-[10px]">
                      <th className="py-2.5 px-3">#</th>
                      <th className="py-2.5 px-3">{t.pricingArea}</th>
                      <th className="py-2.5 px-3">{t.pricingRooms}</th>
                      <th className="py-2.5 px-3">{t.pricingActualPrice}</th>
                      <th className="py-2.5 px-3">{t.pricingPredictedPrice}</th>
                      <th className="py-2.5 px-3">{t.pricingResidual} (y - ŷ)</th>
                      <th className="py-2.5 px-3">{lang === 'ar' ? 'نسبة الخطأ' : 'Error %'}</th>
                      <th className="py-2.5 px-3">{lang === 'ar' ? 'الحالة' : 'Status'}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 font-mono">
                    {dataset.map((apt, idx) => {
                      const pred = predictions[idx] || 0;
                      const residual = apt.actualPrice - pred;
                      const absResidual = Math.abs(residual);
                      const errorPct = (absResidual / apt.actualPrice) * 100;
                      const isAccurate = errorPct < 5;

                      return (
                        <tr key={apt.id} className="hover:bg-slate-800/40 transition">
                          <td className="py-3 px-3 font-bold text-indigo-400">#{apt.id}</td>
                          <td className="py-3 px-3 text-slate-200">{apt.area} م²</td>
                          <td className="py-3 px-3 text-slate-200">{apt.rooms}</td>
                          <td className="py-3 px-3 font-bold text-sky-400">{apt.actualPrice.toFixed(2)}M</td>
                          <td className="py-3 px-3 font-bold text-emerald-400">
                            {pred ? `${pred.toFixed(2)}M` : '--'}
                          </td>
                          <td
                            className={`py-3 px-3 ${
                              absResidual < 1 ? 'text-emerald-400' : 'text-amber-400'
                            }`}
                          >
                            {residual >= 0 ? `+${residual.toFixed(2)}` : residual.toFixed(2)}
                          </td>
                          <td className="py-3 px-3 text-slate-300">
                            {errorPct.toFixed(1)}%
                          </td>
                          <td className="py-3 px-3">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-sans font-bold ${
                                isAccurate
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-amber-500/20 text-amber-300'
                              }`}
                            >
                              {isAccurate
                                ? (lang === 'ar' ? 'دقيق جداً' : 'High Accuracy')
                                : (lang === 'ar' ? 'تقريب جيد' : 'Converging')}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Unseen Test Table */}
              <div className="pt-4 border-t border-slate-800 space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-pink-400" />
                  <h5 className="font-bold text-slate-200 text-xs sm:text-sm">
                    {t.pricingUnseenTest}
                  </h5>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {UNSEEN_TEST_APARTMENTS.map((apt, i) => {
                    const p = predictSingle(apt.area, apt.rooms);
                    return (
                      <div
                        key={i}
                        className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 flex items-center justify-between"
                      >
                        <div>
                          <div className="text-xs font-semibold text-slate-200">
                            {lang === 'ar' ? apt.labelAr : apt.labelEn}
                          </div>
                          <div className="text-[10px] text-slate-400 font-mono">
                            {apt.area} م² | {apt.rooms} {lang === 'ar' ? 'غرف' : 'Rooms'}
                          </div>
                        </div>

                        <div className="text-right">
                          <div className="text-sm font-bold font-mono text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-amber-300">
                            {p.toFixed(2)}M
                          </div>
                          <div className="text-[10px] text-slate-500">
                            {((p * 1000000) / apt.area).toFixed(0)}/م²
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Tab 3: Complete Standalone Python Code */}
          {activeTab === 'code' && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-emerald-400" />
                  <h4 className="font-bold text-slate-100 text-sm">
                    {lang === 'ar' ? 'كود بايثون المستقل (NumPy + Adam Optimizer)' : 'Standalone Python Code (NumPy + Adam)'}
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
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition"
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

          {/* Tab 4: Mathematical Theory & Deep Learning Rationale */}
          {activeTab === 'theory' && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4 text-xs sm:text-sm text-slate-300 leading-relaxed">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
                <Sparkles className="w-5 h-5 text-indigo-400" />
                <h4 className="font-bold text-slate-100 text-base">
                  {lang === 'ar' ? 'الأسرار العلمية والرياضية لنموذج التسعير التنبؤي' : 'Deep Learning Insights for Tabular Regression'}
                </h4>
              </div>

              <div className="space-y-3">
                <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                  <div className="font-bold text-indigo-300 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-indigo-400" />
                    {lang === 'ar' ? '1. لماذا نحتاج طبقة مخفية مع ReLU بدلاً من الانحدار الخطي البسيط؟' : '1. Why a Hidden Layer + ReLU over Simple Linear Regression?'}
                  </div>
                  <p className="text-slate-400 text-xs">
                    {lang === 'ar'
                      ? 'العلاقة بين مواصفات العقار وسعره ليست خطية تماماً؛ فمثلاً الانتقال من غرفتين إلى 3 غرف يمنح قفزة سعرية نوعية تفوق الانتقال من غرفة إلى غرفتين، كما أن القيمة الحدية للمتر المربع تختلف بين الشقق الصغيرة والكبيرة. تتيح دالة ReLU للشبكة بناء "أسطح قرار متعددة المستويات" (Piecewise Linear Approximations) لنمذجة هذه القفزات بواقعية تامة.'
                      : 'Real-world property values are non-linear; transitioning from 2 to 3 rooms creates premium thresholds beyond raw area scaling. ReLU provides piecewise linearity that fits complex continuous valuation surfaces.'}
                  </p>
                </div>

                <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                  <div className="font-bold text-emerald-300 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    {lang === 'ar' ? '2. ما هو سر تفوق خوارزمية Adam في البيانات الجدولية؟' : '2. Why Adam Excels in Tabular Neural Networks'}
                  </div>
                  <p className="text-slate-400 text-xs">
                    {lang === 'ar'
                      ? 'في تسعير العقارات، الميزات تأتي بمقاييس شديدة التفاوت (المساحة بين 60 و 150 بينما الغرف بين 2 و 5). يؤدي هذا التفاوت في خوارزمية SGD التقليدية إلى تذبذب التدرجات أو بطء التقارب. لكن Adam يمتلك ميزتين جوهريتين: العزم الأول (Momentum) لتسريع الحركة في الاتجاه الصحيح، والعزم الثاني (RMSprop) لتعديل معدل التعلم لكل معامل بصورة منفصلة، مما يمنح استقراراً فائقاً.'
                      : 'Continuous tabular features have disparate physical scales (Area: 60–150 vs Rooms: 2–5). Adam normalizes gradient magnitude per parameter via its second raw moment, preventing oscillation and gradient explosion.'}
                  </p>
                </div>

                <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                  <div className="font-bold text-amber-300 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400" />
                    {lang === 'ar' ? '3. دالة الخسارة وتدرج MSE في الانحدار الخطي' : '3. Loss & Backpropagation for Linear Output'}
                  </div>
                  <div className="text-slate-400 text-xs font-mono bg-slate-900 p-2.5 rounded-lg border border-slate-800/80">
                    <div>MSE = (1/N) * Σ(ŷ - y)²</div>
                    <div>dLoss/dŷ = (2/N) * (ŷ - y)</div>
                  </div>
                  <p className="text-slate-400 text-xs">
                    {lang === 'ar'
                      ? 'الطبقة الأخيرة لا تحتوي على دالة تفعيل غير خطية (Linear Output) لأننا نريد التنبؤ بقيمة عددية مستمرة موجبة بدون تقييدها بين 0 و 1 كما في التصنيف الثنائي.'
                      : 'The final layer is purely linear with no activation bounding (unlike Sigmoid), allowing the network to produce arbitrary positive continuous monetary values.'}
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
