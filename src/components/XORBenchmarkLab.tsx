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
  CheckCircle2,
  XCircle,
  Copy,
  Check,
  Download,
  BookOpen,
  Layers,
  Activity,
  Sparkles,
  Info,
  Sliders,
  Cpu,
  ArrowRight,
  TrendingDown,
} from 'lucide-react';
import { DenseLayer } from '../engine/layers';
import {
  Activation_ReLU,
  Activation_Sigmoid,
  Activation_Tanh,
  Activation_LeakyReLU,
  LayerInterface,
} from '../engine/activations';
import { Loss_MSE } from '../engine/loss';
import { Optimizer_SGD, Optimizer_Adam, OptimizerInterface } from '../engine/optimizers';
import { NeuralNetwork } from '../engine/model';
import { MatrixOps } from '../engine/matrix';
import { Language, translations } from '../translations';

interface XORBenchmarkLabProps {
  lang: Language;
}

type ModelArchitectureType = 'mlp' | 'linear';
type ActivationChoice = 'relu' | 'leaky_relu' | 'sigmoid' | 'tanh';
type OptimizerChoice = 'sgd' | 'adam';

// Standard 4 XOR samples
const XOR_X = [
  [0.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [1.0, 1.0],
];
const XOR_Y = [
  [0.0],
  [1.0],
  [1.0],
  [0.0],
];

export const XORBenchmarkLab: React.FC<XORBenchmarkLabProps> = ({ lang }) => {
  const t = translations[lang];

  // Architecture & Settings
  const [archType, setArchType] = useState<ModelArchitectureType>('mlp');
  const [hiddenNeurons, setHiddenNeurons] = useState<number>(4);
  const [activationChoice, setActivationChoice] = useState<ActivationChoice>('relu');
  const [optimizerChoice, setOptimizerChoice] = useState<OptimizerChoice>('sgd');
  const [learningRate, setLearningRate] = useState<number>(0.1);

  // Training state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [currentLoss, setCurrentLoss] = useState<number>(0.25);
  const [predictions, setPredictions] = useState<number[][]>([[0.5], [0.5], [0.5], [0.5]]);
  const [accuracy, setAccuracy] = useState<number>(0.5);

  // Inspector / Code tabs
  const [activeSubView, setActiveSubView] = useState<'visuals' | 'weights' | 'code' | 'theory'>('visuals');
  const [copiedCode, setCopiedCode] = useState(false);
  const [hoverCoord, setHoverCoord] = useState<{ x1: number; x2: number; pred: number } | null>(null);

  // Canvas ref
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationReqRef = useRef<number | null>(null);

  // Build the network model
  const createModel = useCallback(
    (
      architecture: ModelArchitectureType,
      neurons: number,
      act: ActivationChoice,
      opt: OptimizerChoice,
      lr: number
    ): NeuralNetwork => {
      const net = new NeuralNetwork();

      if (architecture === 'linear') {
        // Pure single-layer linear perceptron (no hidden layer)
        const dense = new DenseLayer(2, 1, 0.1);
        net.add(dense);
      } else {
        // Multi-Layer MLP (2 -> hiddenNeurons -> 1)
        const l1 = new DenseLayer(2, neurons, 0.1);
        net.add(l1);

        let actLayer: LayerInterface = new Activation_ReLU();
        if (act === 'leaky_relu') actLayer = new Activation_LeakyReLU(0.01);
        if (act === 'sigmoid') actLayer = new Activation_Sigmoid();
        if (act === 'tanh') actLayer = new Activation_Tanh();
        net.add(actLayer);

        const l2 = new DenseLayer(neurons, 1, 0.1);
        net.add(l2);
      }

      const lossFn = new Loss_MSE();
      const optimizer: OptimizerInterface =
        opt === 'sgd' ? new Optimizer_SGD(lr, 0.0) : new Optimizer_Adam(lr);

      net.set(lossFn, optimizer);
      return net;
    },
    []
  );

  // Persistent model ref
  const modelRef = useRef<NeuralNetwork>(
    createModel(archType, hiddenNeurons, activationChoice, optimizerChoice, learningRate)
  );

  // Reset model helper
  const handleResetModel = useCallback(() => {
    setIsTraining(false);
    if (animationReqRef.current) cancelAnimationFrame(animationReqRef.current);

    const net = createModel(archType, hiddenNeurons, activationChoice, optimizerChoice, learningRate);
    modelRef.current = net;

    const preds = net.forward(XOR_X, false);
    const loss = net.loss_function ? net.loss_function.forward(preds, XOR_Y) : 0.25;

    let correct = 0;
    preds.forEach((p, i) => {
      const predictedClass = p[0] >= 0.5 ? 1 : 0;
      if (predictedClass === XOR_Y[i][0]) correct++;
    });

    setEpoch(0);
    setLossHistory([loss]);
    setCurrentLoss(loss);
    setPredictions(preds);
    setAccuracy(correct / 4);
  }, [archType, hiddenNeurons, activationChoice, optimizerChoice, learningRate, createModel]);

  // Whenever architecture or core config changes, reset model
  useEffect(() => {
    handleResetModel();
  }, [archType, hiddenNeurons, activationChoice, optimizerChoice, handleResetModel]);

  // Learning rate updates directly on optimizer without wiping weights
  useEffect(() => {
    if (modelRef.current.optimizer) {
      modelRef.current.optimizer.learning_rate = learningRate;
    }
  }, [learningRate]);

  // Execute single step
  const executeStep = useCallback((batchCount = 1) => {
    const net = modelRef.current;
    let lastLoss = 0;
    let lastPreds = predictions;

    for (let i = 0; i < batchCount; i++) {
      const stepRes = net.step(XOR_X, XOR_Y);
      lastLoss = stepRes.loss;
      lastPreds = stepRes.prediction;
    }

    let correct = 0;
    lastPreds.forEach((p, i) => {
      const predictedClass = p[0] >= 0.5 ? 1 : 0;
      if (predictedClass === XOR_Y[i][0]) correct++;
    });

    setEpoch(prev => prev + batchCount);
    setCurrentLoss(lastLoss);
    setPredictions(lastPreds);
    setAccuracy(correct / 4);
    setLossHistory(prev => {
      const updated = [...prev, lastLoss];
      return updated.length > 500 ? updated.slice(updated.length - 500) : updated;
    });
  }, [predictions]);

  // Run full 1500 epochs batch
  const handleRun1500Epochs = () => {
    setIsTraining(false);
    executeStep(1500);
  };

  // Live training loop
  useEffect(() => {
    if (!isTraining) return;

    let isMounted = true;
    const loop = () => {
      if (!isMounted) return;
      executeStep(5);
      animationReqRef.current = requestAnimationFrame(loop);
    };

    animationReqRef.current = requestAnimationFrame(loop);

    return () => {
      isMounted = false;
      if (animationReqRef.current) cancelAnimationFrame(animationReqRef.current);
    };
  }, [isTraining, executeStep]);

  // Render 2D Decision Boundary on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Viewport domain: [-0.5, 1.5] for both X1 and X2 (exact user Python meshgrid)
    const xMin = -0.5;
    const xMax = 1.5;
    const yMin = -0.5;
    const yMax = 1.5;

    const res = 60;
    const cellW = width / res;
    const cellH = height / res;

    // Generate grid points
    const gridPoints: number[][] = [];
    for (let j = 0; j < res; j++) {
      const yVal = yMax - (j / (res - 1)) * (yMax - yMin);
      for (let i = 0; i < res; i++) {
        const xVal = xMin + (i / (res - 1)) * (xMax - xMin);
        gridPoints.push([xVal, yVal]);
      }
    }

    // Forward pass grid
    let gridPreds: number[][] = [];
    try {
      gridPreds = modelRef.current.forward(gridPoints, false);
    } catch {
      gridPreds = gridPoints.map(() => [0.5]);
    }

    // Draw background heatmap
    let pIdx = 0;
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const rawVal = gridPreds[pIdx] ? gridPreds[pIdx][0] : 0.5;
        pIdx++;

        // Clamp between 0 and 1
        const prob = Math.max(0, Math.min(1, rawVal));

        // Color gradient:
        // prob < 0.5: Blue/Indigo (Class 0)
        // prob >= 0.5: Emerald/Cyan/Amber (Class 1)
        let r = 0, g = 0, b = 0;
        if (prob < 0.5) {
          const factor = (0.5 - prob) / 0.5; // 0 to 1
          r = Math.round(15 + 15 * factor);
          g = Math.round(23 + 40 * factor);
          b = Math.round(42 + 150 * factor); // Rich slate-indigo
        } else {
          const factor = (prob - 0.5) / 0.5; // 0 to 1
          r = Math.round(16 + 230 * factor);
          g = Math.round(185 + 30 * factor);
          b = Math.round(129 - 40 * factor); // Glowing emerald / amber
        }

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(i * cellW, j * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Draw 0.5 Decision Boundary contour line
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);

    for (let j = 0; j < res - 1; j++) {
      for (let i = 0; i < res - 1; i++) {
        const idxTopLeft = j * res + i;
        const idxTopRight = j * res + (i + 1);
        const idxBotLeft = (j + 1) * res + i;

        const valTL = gridPreds[idxTopLeft] ? gridPreds[idxTopLeft][0] : 0.5;
        const valTR = gridPreds[idxTopRight] ? gridPreds[idxTopRight][0] : 0.5;
        const valBL = gridPreds[idxBotLeft] ? gridPreds[idxBotLeft][0] : 0.5;

        // Check if 0.5 threshold is crossed horizontally
        if ((valTL >= 0.5 && valTR < 0.5) || (valTL < 0.5 && valTR >= 0.5)) {
          const xPos = i * cellW + cellW * 0.5;
          const yPos = j * cellH + cellH * 0.5;
          ctx.beginPath();
          ctx.arc(xPos, yPos, 1.2, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.fill();
        }
        // Check if crossed vertically
        if ((valTL >= 0.5 && valBL < 0.5) || (valTL < 0.5 && valBL >= 0.5)) {
          const xPos = i * cellW + cellW * 0.5;
          const yPos = j * cellH + cellH * 0.5;
          ctx.beginPath();
          ctx.arc(xPos, yPos, 1.2, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.fill();
        }
      }
    }
    ctx.setLineDash([]);

    // Draw coordinate axes at X=0 and Y=0
    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height;

    const zeroX = toCanvasX(0);
    const zeroY = toCanvasY(0);
    const oneX = toCanvasX(1);
    const oneY = toCanvasY(1);

    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(zeroX, 0);
    ctx.lineTo(zeroX, height);
    ctx.moveTo(0, zeroY);
    ctx.lineTo(width, zeroY);
    ctx.stroke();

    // Unit square outline [0, 1] x [0, 1]
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(zeroX, oneY, oneX - zeroX, zeroY - oneY);
    ctx.setLineDash([]);

    // Draw the 4 XOR sample points
    XOR_X.forEach((pt, i) => {
      const px = toCanvasX(pt[0]);
      const py = toCanvasY(pt[1]);
      const trueVal = XOR_Y[i][0];
      const curPred = predictions[i] ? predictions[i][0] : 0.5;
      const isCorrect = (curPred >= 0.5 ? 1 : 0) === trueVal;

      // Outer glow ring
      ctx.beginPath();
      ctx.arc(px, py, 14, 0, Math.PI * 2);
      ctx.fillStyle = isCorrect ? 'rgba(34, 197, 94, 0.35)' : 'rgba(239, 68, 68, 0.4)';
      ctx.fill();

      // Main circle
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.fillStyle = trueVal === 1 ? '#10b981' : '#3b82f6';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.fill();
      ctx.stroke();

      // Text label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = pt[0] === 1 ? 'right' : 'left';
      ctx.textBaseline = 'middle';
      const offset = pt[0] === 1 ? -18 : 18;
      ctx.fillText(`(${pt[0]}, ${pt[1]}) = ${trueVal}`, px + offset, py);
    });
  }, [predictions, epoch]);

  // Handle canvas mouse hover to inspect continuous plane
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const xMin = -0.5, xMax = 1.5, yMin = -0.5, yMax = 1.5;
    const x1 = xMin + (cx / rect.width) * (xMax - xMin);
    const x2 = yMax - (cy / rect.height) * (yMax - yMin);

    try {
      const pred = modelRef.current.forward([[x1, x2]], false);
      setHoverCoord({
        x1: Number(x1.toFixed(3)),
        x2: Number(x2.toFixed(3)),
        pred: pred[0] ? Number(pred[0][0].toFixed(4)) : 0.5,
      });
    } catch {
      setHoverCoord(null);
    }
  };

  // Standalone Python code matching user snippet
  const pythonScriptCode = `import numpy as np
import matplotlib.pyplot as plt

# --- 1. بناء المكونات الأساسية لمنظومة MiniDeepEngine ---
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # تهيئة الأوزان والانحياز كما في المعمارية
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        # تطبيق قاعدة السلسلة (Chain Rule) لحساب مشتقات الأوزان والانحياز
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

# --- 2. إعداد بيانات مشكلة XOR غير الخطية الكلاسيكية ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]]) # الناتج المنطقي لـ XOR

# --- 3. بناء الشبكة (طبقتين: مخفية 4 خلايا مع ReLU + طبقة مخرجات) ---
hidden_layer = DenseLayer(n_inputs=2, n_neurons=4)
activation = Activation_ReLU()
output_layer = DenseLayer(n_inputs=4, n_neurons=1)
loss_fn = Loss_MSE()

epochs = 1500
learning_rate = 0.1
loss_history = []

print("=== بدء تدريب شبكة MiniDeepEngine على مشكلة XOR ===")

for epoch in range(epochs):
    # الممر الأمامي (Forward Pass)
    dense1_out = hidden_layer.forward(X)
    act_out = activation.forward(dense1_out)
    preds = output_layer.forward(act_out)
    
    # حساب الخطأ (MSE Loss)
    loss = loss_fn.forward(preds, y)
    loss_history.append(loss)
    
    # الانتشار الخلفي وقاعدة السلسلة (Backpropagation & Chain Rule)
    d_loss = loss_fn.backward(preds, y)
    output_layer.backward(d_loss)
    activation.backward(output_layer.dinputs)
    hidden_layer.backward(activation.dvalues)
    
    # تحديث الأوزان عبر Gradient Descent البسيط
    output_layer.weights -= learning_rate * output_layer.dweights
    output_layer.biases -= learning_rate * output_layer.dbiases
    hidden_layer.weights -= learning_rate * hidden_layer.dweights
    hidden_layer.biases -= learning_rate * hidden_layer.dbiases

    if (epoch + 1) % 300 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss:.6f}")

# --- 4. اختبار التوقعات النهائية لجدول الحقيقة ---
print("\\n--- التوقعات النهائية لجدول XOR بعد 1500 دورة ---")
final_preds = output_layer.forward(activation.forward(hidden_layer.forward(X)))
for i in range(len(X)):
    pred_val = final_preds[i][0]
    pred_cls = 1 if pred_val >= 0.5 else 0
    print(f"المدخل: {X[i]} -> المستهدف: {y[i][0]} -> التنبؤ: {pred_val:.4f} (الفئة: {pred_cls})")

# --- 5. رسم منحنى الخسارة وخريطة القرار (Decision Boundary) ---
plt.figure(figsize=(12, 5))

# رسم منحنى الخسارة
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='#4f46e5', linewidth=2)
plt.title("منحنى الخسارة (Loss Curve) لمشكلة XOR")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.6)

# رسم خريطة القرار (Decision Boundary)
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_preds = output_layer.forward(activation.forward(hidden_layer.forward(grid_points)))
grid_preds = grid_preds.reshape(xx.shape)

plt.contourf(xx, yy, grid_preds, levels=50, cmap='coolwarm', alpha=0.8)
plt.colorbar(label='Output Prediction (Probability)')
plt.contour(xx, yy, grid_preds, levels=[0.5], colors='black', linewidths=2)

for i in range(len(X)):
    color = 'red' if y[i][0] == 0 else 'blue'
    plt.scatter(X[i][0], X[i][1], color=color, s=160, edgecolors='black', linewidth=2)
    plt.text(X[i][0] + 0.04, X[i][1] + 0.04, f"({X[i][0]},{X[i][1]})={y[i][0]}", fontsize=10, weight='bold')

plt.title("خريطة القرار لمشكلة XOR (Decision Boundary)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.tight_layout()
plt.show()
`;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(pythonScriptCode);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleDownloadPython = () => {
    const blob = new Blob([pythonScriptCode], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'minideepengine_xor_benchmark.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Inspect weights and gradients
  const layers = modelRef.current.layers;
  const denseL1 = layers[0] instanceof DenseLayer ? (layers[0] as DenseLayer) : null;
  const denseL2 = layers[2] instanceof DenseLayer ? (layers[2] as DenseLayer) : (layers[1] instanceof DenseLayer ? (layers[1] as DenseLayer) : null);

  return (
    <div className="space-y-6">
      {/* Top Banner & Architectural Paradigm Selector */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 rounded-2xl p-5 shadow-2xl relative overflow-hidden">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2.5">
              <span className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <Cpu className="w-5 h-5" />
              </span>
              <h2 className="text-xl sm:text-2xl font-black text-white tracking-wide">
                {t.xorBenchmarkTab}
              </h2>
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                Minsky 1969 vs Rumelhart 1986
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-300 max-w-3xl leading-relaxed">
              {t.xorSubtitle}
            </p>
          </div>

          {/* Quick Paradigm Comparison Switcher */}
          <div className="bg-slate-950/80 p-1.5 rounded-xl border border-slate-800 flex items-center gap-1 self-stretch lg:self-auto">
            <button
              onClick={() => setArchType('linear')}
              className={`flex-1 sm:flex-initial px-3.5 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                archType === 'linear'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-lg'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
              <span>البرسبترون الخطي (Linear)</span>
            </button>
            <button
              onClick={() => setArchType('mlp')}
              className={`flex-1 sm:flex-initial px-3.5 py-2 rounded-lg text-xs font-bold transition flex items-center justify-center gap-2 ${
                archType === 'mlp'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-lg'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>شبكة متعددة الطبقات (2-Layer MLP + ReLU)</span>
            </button>
          </div>
        </div>

        {/* Dynamic Architectural Insight Badge */}
        <div className="mt-4 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
          {archType === 'linear' ? (
            <div className="flex items-center gap-2 text-rose-300 bg-rose-950/40 px-3 py-1.5 rounded-lg border border-rose-800/40">
              <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>
                <strong>معضلة الخطوط المستقيمة:</strong> يستحيل رياضياً رسم خط مستقيم واحد يفصل (0,1) و (1,0) عن (0,0) و (1,1). ستظل الخسارة عالقة عند 0.25 والدقة 50%.
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-emerald-300 bg-emerald-950/40 px-3 py-1.5 rounded-lg border border-emerald-800/40">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                <strong>قوة الطبقة المخفية ودالة ReLU:</strong> تقوم العصبونات الأربعة بطي الفضاء الإحداثي (Space Folding) بحيث تصبح نقاط XOR قابلة للفصل خطياً في الفضاء الرباعي!
              </span>
            </div>
          )}

          <div className="flex items-center gap-2 text-slate-400 text-[11px] font-mono">
            <span>الهدف: (0,0)=0 | (0,1)=1 | (1,0)=1 | (1,1)=0</span>
          </div>
        </div>
      </div>

      {/* Main Control HUD */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-xl flex flex-wrap items-center justify-between gap-4">
        {/* Playback & Step Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setIsTraining(!isTraining)}
            className={`px-5 py-2.5 rounded-xl font-bold text-xs sm:text-sm flex items-center gap-2 transition shadow-lg ${
              isTraining
                ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-600/30'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30'
            }`}
          >
            {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-white" />}
            <span>{isTraining ? t.pause : t.train}</span>
          </button>

          <button
            onClick={() => executeStep(1)}
            disabled={isTraining}
            className="px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs sm:text-sm font-semibold flex items-center gap-1.5 transition disabled:opacity-50"
            title="تنفيذ دورة تدريبية واحدة"
          >
            <StepForward className="w-4 h-4" />
            <span>خطوة (1 Epoch)</span>
          </button>

          <button
            onClick={() => executeStep(100)}
            disabled={isTraining}
            className="px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs sm:text-sm font-semibold flex items-center gap-1.5 transition disabled:opacity-50"
            title="تنفيذ 100 دورة تدريبية سريعة"
          >
            <FastForward className="w-4 h-4" />
            <span>+100 دورة</span>
          </button>

          <button
            onClick={handleRun1500Epochs}
            disabled={isTraining}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs sm:text-sm font-bold flex items-center gap-1.5 transition shadow-lg shadow-emerald-600/20 disabled:opacity-50"
            title="تشغيل كامل الـ 1500 دورة المحددة في كود بايثون"
          >
            <Sparkles className="w-4 h-4 text-emerald-200" />
            <span>{t.xorRun1500}</span>
          </button>

          <button
            onClick={handleResetModel}
            className="px-3.5 py-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 text-xs sm:text-sm font-semibold flex items-center gap-1.5 transition"
            title="إعادة ضبط أوزان النموذج عشوائياً"
          >
            <RotateCcw className="w-4 h-4" />
            <span>{t.reset}</span>
          </button>
        </div>

        {/* Live Metrics Counters */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="bg-slate-950 px-3.5 py-2 rounded-xl border border-slate-800 flex items-center gap-2">
            <span className="text-[11px] text-slate-400 font-medium">الدورات (Epochs):</span>
            <span className="font-mono font-bold text-sm text-indigo-300">{epoch}</span>
          </div>

          <div className="bg-slate-950 px-3.5 py-2 rounded-xl border border-slate-800 flex items-center gap-2">
            <span className="text-[11px] text-slate-400 font-medium">الخسارة (MSE):</span>
            <span className="font-mono font-bold text-sm text-amber-300">
              {currentLoss.toFixed(6)}
            </span>
          </div>

          <div className="bg-slate-950 px-3.5 py-2 rounded-xl border border-slate-800 flex items-center gap-2">
            <span className="text-[11px] text-slate-400 font-medium">الدقة (Accuracy):</span>
            <span
              className={`font-mono font-bold text-sm ${
                accuracy === 1.0 ? 'text-emerald-400' : accuracy >= 0.75 ? 'text-amber-400' : 'text-rose-400'
              }`}
            >
              {(accuracy * 100).toFixed(0)}% ({Math.round(accuracy * 4)}/4)
            </span>
          </div>
        </div>
      </div>

      {/* Hyperparameter Settings Drawer */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-4">
          {/* Learning Rate */}
          <div className="flex items-center gap-2">
            <span className="text-slate-400">معدل التعلم (LR):</span>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={learningRate}
              onChange={e => setLearningRate(parseFloat(e.target.value))}
              className="w-24 accent-indigo-500 cursor-pointer"
            />
            <span className="font-mono text-indigo-300 font-bold">{learningRate.toFixed(2)}</span>
          </div>

          {/* Optimizer Switch */}
          <div className="flex items-center gap-2 border-r border-slate-800 pr-4">
            <span className="text-slate-400">المحسّن (Optimizer):</span>
            <select
              value={optimizerChoice}
              onChange={e => setOptimizerChoice(e.target.value as OptimizerChoice)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 font-mono text-xs focus:outline-none focus:border-indigo-500"
            >
              <option value="sgd">SGD (Gradient Descent كود بايثون)</option>
              <option value="adam">Adam (تكيفي فائق السرعة)</option>
            </select>
          </div>

          {/* Hidden Neurons (if MLP) */}
          {archType === 'mlp' && (
            <div className="flex items-center gap-2 border-r border-slate-800 pr-4">
              <span className="text-slate-400">خلايا الطبقة المخفية:</span>
              <div className="flex gap-1">
                {[2, 4, 8].map(n => (
                  <button
                    key={n}
                    onClick={() => setHiddenNeurons(n)}
                    className={`px-2 py-0.5 rounded font-mono text-xs ${
                      hiddenNeurons === n
                        ? 'bg-indigo-600 text-white font-bold'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Activation (if MLP) */}
          {archType === 'mlp' && (
            <div className="flex items-center gap-2 border-r border-slate-800 pr-4">
              <span className="text-slate-400">دالة التفعيل:</span>
              <select
                value={activationChoice}
                onChange={e => setActivationChoice(e.target.value as ActivationChoice)}
                className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200 font-mono text-xs focus:outline-none focus:border-indigo-500"
              >
                <option value="relu">ReLU (الافتراضية في الكود)</option>
                <option value="leaky_relu">LeakyReLU</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
              </select>
            </div>
          )}
        </div>

        <div className="text-[11px] text-slate-400 italic">
          💡 يمكنك النقر على الخريطة لتفقد التنبؤات في أي نقطة إحداثية
        </div>
      </div>

      {/* Sub-Tabs: Visuals / Weights / Code / Theory */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveSubView('visuals')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
            activeSubView === 'visuals'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>خريطة القرار وجدول الحقيقة (Decision Boundary & Truth Table)</span>
        </button>

        <button
          onClick={() => setActiveSubView('weights')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
            activeSubView === 'weights'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>{t.xorWeightsChainRule}</span>
        </button>

        <button
          onClick={() => setActiveSubView('code')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
            activeSubView === 'code'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <BookOpen className="w-4 h-4" />
          <span>{t.xorPythonCode}</span>
        </button>

        <button
          onClick={() => setActiveSubView('theory')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold transition ${
            activeSubView === 'theory'
              ? 'bg-indigo-600 text-white shadow-md'
              : 'text-slate-400 hover:text-white hover:bg-slate-900'
          }`}
        >
          <Sparkles className="w-4 h-4 text-amber-400" />
          <span>التاريخ والرياضيات وراء معضلة XOR</span>
        </button>
      </div>

      {/* VIEW 1: VISUALS (Decision Boundary Canvas + Truth Table + Loss Curve) */}
      {activeSubView === 'visuals' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: 2D Decision Boundary Canvas (7 Cols) */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
                <h3 className="text-sm font-bold text-white">{t.xorDecisionBoundary}</h3>
                <span className="text-[11px] font-mono text-slate-400">[-0.5, 1.5]²</span>
              </div>

              {hoverCoord && (
                <div className="text-[11px] font-mono bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800 text-indigo-300">
                  X1: {hoverCoord.x1} | X2: {hoverCoord.x2} → P(ŷ=1): {hoverCoord.pred}
                </div>
              )}
            </div>

            {/* Canvas Container */}
            <div className="relative aspect-square max-h-[420px] mx-auto w-full bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-inner flex items-center justify-center">
              <canvas
                ref={canvasRef}
                width={400}
                height={400}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={() => setHoverCoord(null)}
                className="w-full h-full object-contain cursor-crosshair"
              />
            </div>

            {/* Legend */}
            <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-3">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-[#3b82f6] border border-white" />
                  <span>الفئة 0 (أزرق داكن)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-[#10b981] border border-white" />
                  <span>الفئة 1 (أخضر زمردي)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-4 h-0.5 border-t-2 border-dashed border-white" />
                  <span>حد القرار (ŷ = 0.5)</span>
                </div>
              </div>

              <div className="text-[11px] font-mono text-slate-400">
                الحلقة الخارجية: أخضر (صحيح) / أحمر (خاطئ)
              </div>
            </div>
          </div>

          {/* Right Column: Truth Table & Loss Curve (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* Live Truth Table Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-bold text-white">{t.xorTruthTable}</h3>
                </div>
                <span className="text-xs font-mono font-bold text-indigo-400">
                  {Math.round(accuracy * 4)} / 4 حالات صحيحة
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-right font-mono">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-800/60 text-[11px]">
                      <th className="py-1.5 px-2">X1, X2</th>
                      <th className="py-1.5 px-2">الهدف Y</th>
                      <th className="py-1.5 px-2">التنبؤ ŷ</th>
                      <th className="py-1.5 px-2">الفئة</th>
                      <th className="py-1.5 px-2">الخطأ التربيعي</th>
                      <th className="py-1.5 px-2 text-center">الحالة</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40">
                    {XOR_X.map((pt, i) => {
                      const trueVal = XOR_Y[i][0];
                      const predVal = predictions[i] ? predictions[i][0] : 0.5;
                      const predClass = predVal >= 0.5 ? 1 : 0;
                      const isCorrect = predClass === trueVal;
                      const sqError = Math.pow(predVal - trueVal, 2);

                      return (
                        <tr
                          key={i}
                          className={`hover:bg-slate-800/40 transition ${
                            isCorrect ? 'text-slate-200' : 'bg-rose-950/20 text-rose-200'
                          }`}
                        >
                          <td className="py-2.5 px-2 font-bold text-indigo-300">
                            ({pt[0]}, {pt[1]})
                          </td>
                          <td className="py-2.5 px-2">
                            <span
                              className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                                trueVal === 1
                                  ? 'bg-emerald-500/20 text-emerald-300'
                                  : 'bg-blue-500/20 text-blue-300'
                              }`}
                            >
                              {trueVal}
                            </span>
                          </td>
                          <td className="py-2.5 px-2 text-amber-300 font-bold">
                            {predVal.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-2 font-semibold">
                            {predClass}
                          </td>
                          <td className="py-2.5 px-2 text-slate-400 text-[11px]">
                            {sqError.toFixed(5)}
                          </td>
                          <td className="py-2.5 px-2 text-center">
                            {isCorrect ? (
                              <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 font-bold bg-emerald-950/40 px-2 py-0.5 rounded-full border border-emerald-800/40">
                                <Check className="w-3 h-3" /> PASS
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-[11px] text-rose-400 font-bold bg-rose-950/40 px-2 py-0.5 rounded-full border border-rose-800/40">
                                <XCircle className="w-3 h-3" /> FAIL
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Loss Curve Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <div className="flex items-center gap-2">
                  <TrendingDown className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-bold text-white">{t.xorLossCurve}</h3>
                </div>
                <span className="text-xs font-mono text-amber-300">MSE: {currentLoss.toFixed(6)}</span>
              </div>

              {/* Loss Curve SVG */}
              <div className="h-40 w-full bg-slate-950 rounded-xl p-2 relative overflow-hidden border border-slate-800">
                {lossHistory.length < 2 ? (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500 font-mono">
                    ابدأ التدريب أو انقر تشغيل لرسم منحنى الخسارة...
                  </div>
                ) : (
                  <svg className="w-full h-full overflow-visible" viewBox="0 0 400 120" preserveAspectRatio="none">
                    <defs>
                      <linearGradient id="lossGradientXOR" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#4f46e5" stopOpacity="0.4" />
                        <stop offset="100%" stopColor="#4f46e5" stopOpacity="0.0" />
                      </linearGradient>
                    </defs>

                    {/* Grid lines */}
                    <line x1="0" y1="30" x2="400" y2="30" stroke="#1e293b" strokeDasharray="3 3" />
                    <line x1="0" y1="60" x2="400" y2="60" stroke="#1e293b" strokeDasharray="3 3" />
                    <line x1="0" y1="90" x2="400" y2="90" stroke="#1e293b" strokeDasharray="3 3" />

                    {/* Threshold line at 0.25 (linear limitation) */}
                    <line x1="0" y1="40" x2="400" y2="40" stroke="#f43f5e" strokeWidth="1" strokeDasharray="4 4" />

                    {/* Loss path */}
                    {(() => {
                      const maxVal = Math.max(0.35, ...lossHistory);
                      const points = lossHistory.map((val, idx) => {
                        const x = (idx / (lossHistory.length - 1)) * 400;
                        const y = 115 - (val / maxVal) * 110;
                        return `${x},${Math.max(5, Math.min(115, y))}`;
                      });

                      const pathD = `M ${points.join(' L ')}`;
                      const areaD = `M ${points[0]} L ${points.join(' L ')} L 400,120 L 0,120 Z`;

                      return (
                        <>
                          <path d={areaD} fill="url(#lossGradientXOR)" />
                          <path d={pathD} fill="none" stroke="#6366f1" strokeWidth="2.5" />
                        </>
                      );
                    })()}
                  </svg>
                )}

                {/* Milestone overlay */}
                <div className="absolute bottom-2 left-2 text-[10px] font-mono text-slate-500">
                  خط أحمر منقط: حد عجز البرسبترون الخطي (0.25)
                </div>
              </div>

              {/* Milestones log */}
              <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Epoch 300:</span>
                  <span className="text-slate-300">
                    {epoch >= 300 ? (lossHistory[Math.min(300, lossHistory.length - 1)]?.toFixed(6) || '—') : 'قيد الانتظار'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Epoch 600:</span>
                  <span className="text-slate-300">
                    {epoch >= 600 ? (lossHistory[Math.min(600, lossHistory.length - 1)]?.toFixed(6) || '—') : 'قيد الانتظار'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Epoch 1500:</span>
                  <span className="text-emerald-400 font-bold">
                    {epoch >= 1500 ? (lossHistory[lossHistory.length - 1]?.toFixed(6) || '—') : 'قيد الانتظار'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: WEIGHTS & GRADIENTS (Math Anatomy) */}
      {activeSubView === 'weights' && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2.5">
              <Layers className="w-5 h-5 text-indigo-400" />
              <div>
                <h3 className="text-base font-bold text-white">تشريح الأوزان وتدرجات قاعدة السلسلة (Chain Rule Anatomy)</h3>
                <p className="text-xs text-slate-400">
                  مصفوفات الأوزان والانحياز والتدرجات المشتقة لحظياً أثناء الانتشار الخلفي لمعضلة XOR
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Layer 1: Dense Layer (2 -> 4) */}
            {denseL1 && (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <span className="font-bold text-xs text-indigo-300">
                    الطبقة الأولى: DenseLayer(2, {denseL1.n_neurons})
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    W1: {denseL1.n_inputs}×{denseL1.n_neurons} | b1: 1×{denseL1.n_neurons}
                  </span>
                </div>

                {/* Weights Matrix */}
                <div className="space-y-1">
                  <div className="text-[11px] font-medium text-slate-400">مصفوفة الأوزان W1:</div>
                  <div className="bg-slate-900 p-2 rounded-lg font-mono text-xs overflow-x-auto space-y-1">
                    {denseL1.weights.map((row, r) => (
                      <div key={r} className="flex gap-2">
                        <span className="text-slate-500 w-6">x{r + 1}:</span>
                        {row.map((val, c) => (
                          <span
                            key={c}
                            className={`px-1.5 py-0.5 rounded text-[11px] ${
                              val >= 0 ? 'text-emerald-300 bg-emerald-950/40' : 'text-rose-300 bg-rose-950/40'
                            }`}
                          >
                            {val.toFixed(4)}
                          </span>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Biases Vector */}
                <div className="space-y-1">
                  <div className="text-[11px] font-medium text-slate-400">متجه الانحياز b1:</div>
                  <div className="bg-slate-900 p-2 rounded-lg font-mono text-xs overflow-x-auto flex gap-2">
                    {denseL1.biases[0]?.map((b, c) => (
                      <span key={c} className="text-indigo-300 bg-indigo-950/40 px-1.5 py-0.5 rounded text-[11px]">
                        {b.toFixed(4)}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Gradients dW1 */}
                {denseL1.dweights && denseL1.dweights.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[11px] font-medium text-slate-400">تدرجات الأوزان (dW1 = X^T · δ):</div>
                    <div className="bg-slate-900 p-2 rounded-lg font-mono text-xs overflow-x-auto space-y-1">
                      {denseL1.dweights.map((row, r) => (
                        <div key={r} className="flex gap-2">
                          <span className="text-slate-500 w-6">∂L/∂w:</span>
                          {row.map((val, c) => (
                            <span key={c} className="text-amber-300 text-[11px]">
                              {val.toFixed(5)}
                            </span>
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Layer 2: Output Dense Layer (4 -> 1) */}
            {denseL2 && (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <span className="font-bold text-xs text-emerald-300">
                    طبقة المخرجات: DenseLayer({denseL2.n_inputs}, 1)
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    W2: {denseL2.n_inputs}×1 | b2: 1×1
                  </span>
                </div>

                {/* Weights Matrix */}
                <div className="space-y-1">
                  <div className="text-[11px] font-medium text-slate-400">مصفوفة الأوزان W2:</div>
                  <div className="bg-slate-900 p-2 rounded-lg font-mono text-xs overflow-x-auto space-y-1">
                    {denseL2.weights.map((row, r) => (
                      <div key={r} className="flex items-center justify-between">
                        <span className="text-slate-500 text-[11px]">العصبون المخفي h{r + 1}:</span>
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            row[0] >= 0 ? 'text-emerald-300 bg-emerald-950/40' : 'text-rose-300 bg-rose-950/40'
                          }`}
                        >
                          {row[0].toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Biases */}
                <div className="space-y-1">
                  <div className="text-[11px] font-medium text-slate-400">انحياز المخرج b2:</div>
                  <div className="bg-slate-900 p-2 rounded-lg font-mono text-xs text-indigo-300 font-bold">
                    b2 = {denseL2.biases[0]?.[0]?.toFixed(4) || '0.0000'}
                  </div>
                </div>

                {/* Gradients dW2 */}
                {denseL2.dweights && denseL2.dweights.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[11px] font-medium text-slate-400">تدرجات أوزان المخرجات (dW2 = a1^T · dLoss):</div>
                    <div className="bg-slate-900 p-2 rounded-lg font-mono text-xs overflow-x-auto space-y-1">
                      {denseL2.dweights.map((row, r) => (
                        <div key={r} className="flex items-center justify-between text-amber-300 text-[11px]">
                          <span className="text-slate-500">∂L/∂w2_{r + 1}:</span>
                          <span>{row[0].toFixed(5)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Chain Rule Formula Explanation */}
          <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-2">
            <h4 className="font-bold text-indigo-300 text-sm flex items-center gap-2">
              <Activity className="w-4 h-4" />
              كيف تتنقل الإشارة عبر قاعدة السلسلة (Chain Rule) في هذا النموذج؟
            </h4>
            <div className="font-mono bg-slate-900 p-3 rounded-lg text-slate-200 text-[11px] leading-relaxed space-y-1">
              <div>1. ∂L/∂ŷ = 2 · (ŷ - y) / N   [مشتقة دالة MSE]</div>
              <div>2. ∂L/∂W2 = a1^T · (∂L/∂ŷ)  [تدرج أوزان طبقة المخرجات]</div>
              <div>3. ∂L/∂a1 = (∂L/∂ŷ) · W2^T  [الخطأ العائد إلى الطبقة المخفية]</div>
              <div>4. ∂L/∂z1 = (∂L/∂a1) ⊙ ReLU'(z1)  [حيث ReLU'(z) = 1 إذا كان z &gt; 0 وإلا 0]</div>
              <div>5. ∂L/∂W1 = X^T · (∂L/∂z1)   [تدرج أوزان الطبقة الأولى المدخلة]</div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 3: PYTHON CODE (NumPy + Matplotlib) */}
      {activeSubView === 'code' && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-indigo-400" />
                <span>كود بايثون المستقل (NumPy + Matplotlib) لمشكلة XOR</span>
              </h3>
              <p className="text-xs text-slate-400">
                يمكنك نسخ هذا الكود وتشغيله مباشرة في Jupyter Notebook أو أي بيئة بايثون لتدريب المنظومة ورسم خريطة القرار
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleCopyCode}
                className="px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs flex items-center gap-1.5 transition shadow-lg"
              >
                {copiedCode ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                <span>{copiedCode ? 'تم النسخ!' : 'نسخ الكود'}</span>
              </button>

              <button
                onClick={handleDownloadPython}
                className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs flex items-center gap-1.5 transition"
              >
                <Download className="w-4 h-4" />
                <span>تحميل ملف xor_benchmark.py</span>
              </button>
            </div>
          </div>

          {/* Syntax Highlighted Code Box */}
          <div className="relative bg-slate-950 rounded-xl p-4 border border-slate-800 font-mono text-xs overflow-x-auto text-slate-300 max-h-[500px]">
            <pre className="leading-relaxed whitespace-pre">{pythonScriptCode}</pre>
          </div>
        </div>
      )}

      {/* VIEW 4: THEORY & HISTORY (Minsky & Papert 1969) */}
      {activeSubView === 'theory' && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-5">
          <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
            <div className="p-3 bg-amber-500/20 text-amber-400 rounded-xl border border-amber-500/30">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-black text-white">
                لماذا غيرت مشكلة XOR تاريخ الذكاء الاصطناعي إلى الأبد؟
              </h3>
              <p className="text-xs text-slate-400">
                من شتاء الذكاء الاصطناعي عام 1969 إلى الثورة العميقة لشبكات التعلم الحديثة
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs sm:text-sm text-slate-300 leading-relaxed">
            {/* 1969 Crisis */}
            <div className="bg-slate-950 p-5 rounded-xl border border-rose-900/40 space-y-3">
              <div className="flex items-center gap-2 text-rose-400 font-bold text-sm">
                <XCircle className="w-4 h-4" />
                <span>1969: كتاب "Perceptrons" وصدمة مينسكي وبابيرت</span>
              </div>
              <p>
                نشر العالمان <strong>مارفن مينسكي</strong> و<strong>سيمور بابيرت</strong> (من معهد MIT) كتابهما الشهير الذي أثبت رياضياً أن البرسبترون الخطي (Linear Perceptron) غير قادر جوهرياً على تمثيل دالة XOR البسيطة.
              </p>
              <p>
                لأن الدالة تتطلب فصلاً قطرياً: النقطتان (0,1) و (1,0) تأخذان القيمة 1، بينما النقطتان (0,0) و (1,1) تأخذان 0. ولا يوجد أي خط مستقيم في الفضاء الثنائي يمكنه عزل هذين الزوجين معاً! أدى هذا الاستنتاج إلى انقطاع التمويل وتوقف أبحاث الشبكات العصبية لما يقارب 15 عاماً فيما عُرف بـ <strong>شتاء الذكاء الاصطناعي الأول</strong>.
              </p>
            </div>

            {/* 1986 Breakthrough */}
            <div className="bg-slate-950 p-5 rounded-xl border border-emerald-900/40 space-y-3">
              <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                <CheckCircle2 className="w-4 h-4" />
                <span>1986: الخوارزمية العبقرية (Backpropagation)</span>
              </div>
              <p>
                أعاد العلماء <strong>ديفيد روميلهارت</strong>، <strong>جيفري هينتون</strong>، و<strong>رونالد ويليامز</strong> إحياء المجال عبر ورقتهم التاريخية في مجلة Nature، حيث أثبتوا أن إضافة <strong>طبقة مخفية (Hidden Layer)</strong> مع دالة تفعيل غير خطية (Non-linear Activation) كفيلة بحل معضلة XOR بالكامل.
              </p>
              <p>
                السر يكمن في ما يسمى <strong>"طي الفضاء الإحداثي" (Space Folding)</strong>: تقوم خلايا الطبقة المخفية بتحويل الإحداثيات الثنائية المسطحة إلى فضاء أعلى أبعاداً (مثل الفضاء الرباعي عند استخدام 4 خلايا)، وفي هذا الفضاء تصبح النقاط قابلة للفصل بمستوٍ خطي بمنتهى السهولة!
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
