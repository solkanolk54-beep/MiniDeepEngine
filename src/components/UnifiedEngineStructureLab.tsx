/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  FolderTree,
  FileCode,
  Copy,
  Check,
  Download,
  Package,
  Layers,
  Sparkles,
  Terminal,
  Activity,
  Cpu,
  ShieldCheck,
  Play,
  RotateCcw,
} from 'lucide-react';
import { Language } from '../translations';

interface UnifiedEngineStructureLabProps {
  lang: Language;
}

interface EngineFile {
  name: string;
  path: string;
  descriptionAr: string;
  descriptionEn: string;
  badge: string;
  exports: string[];
  code: string;
}

const ENGINE_FILES: EngineFile[] = [
  {
    name: '__init__.py',
    path: 'minideep_engine/__init__.py',
    descriptionAr: 'ملف التهيئة الموحد للحزمة - يتيح الاستدعاء المباشر النظيف لكافة المكونات الأساسية بنقرة واحدة.',
    descriptionEn: 'Unified package init file - provides top-level clean imports for all engine components.',
    badge: 'Package Entrypoint',
    exports: ['DenseLayer', 'Activation_ReLU', 'Activation_Sigmoid', 'Loss_MSE', 'Loss_BinaryCrossEntropy', 'Optimizer_Adam', 'Optimizer_SGD', 'save_model', 'load_model'],
    code: `"""
MiniDeepEngine - Deep Learning Framework from Scratch
منظومة تعلم عميق وشبكات عصبية مبنية من الصفر باستخدام الرياضيات الصرفة ومكتبة NumPy فقط.

الهيكل الموحد للحزمة:
- layers: الطبقات العصبية (DenseLayer)
- activations: دوال التفعيل الرياضية ومشتقاتها (ReLU, Sigmoid)
- losses: دوال قياس الخسارة ومشتقاتها (Loss_MSE, Loss_BinaryCrossEntropy)
- optimizers: خوارزميات تحسين الأوزان (Optimizer_Adam, Optimizer_SGD)
- utils: دوال حفظ واسترجاع النماذج بصيغة JSON خفيفة (save_model, load_model)
"""

from .layers import DenseLayer
from .activations import Activation_ReLU, Activation_Sigmoid
from .losses import Loss_MSE, Loss_BinaryCrossEntropy
from .optimizers import Optimizer_Adam, Optimizer_SGD
from .utils import save_model, load_model

__version__ = "1.0.0"
__author__ = "MiniDeepEngine Team"

__all__ = [
    "DenseLayer",
    "Activation_ReLU",
    "Activation_Sigmoid",
    "Loss_MSE",
    "Loss_BinaryCrossEntropy",
    "Optimizer_Adam",
    "Optimizer_SGD",
    "save_model",
    "load_model",
]
`,
  },
  {
    name: 'layers.py',
    path: 'minideep_engine/layers.py',
    descriptionAr: 'طبقات الشبكة العصبية كاملة الاتصال (DenseLayer) مع التمرير الأمامي وتطبيق قاعدة السلسلة (Chain Rule) في التمرير العكسي.',
    descriptionEn: 'Fully connected dense layers with forward matrix multiplication and backward gradient propagation.',
    badge: 'Core Compute',
    exports: ['DenseLayer'],
    code: `"""
layers.py - طبقات الشبكة العصبية (Neural Network Layers)
منظومة MiniDeepEngine
"""

import numpy as np


class DenseLayer:
    """
    الطبقة كاملة الاتصال (Dense / Fully Connected Layer)
    
    المعادلة الأمامية (Forward Pass):
        Z = X · W + B
        
    المعادلة العكسية (Backward Pass / Chain Rule):
        ∂L/∂W = X^T · ∂L/∂Z
        ∂L/∂B = Σ(∂L/∂Z)
        ∂L/∂X = ∂L/∂Z · W^T
    """
    def __init__(self, n_inputs: int, n_neurons: int, weight_scale: float = 0.01):
        """
        تهيئة أوزان الطبقة وانحيازاتها
        :param n_inputs: عدد الميزات أو المدخلات القادمة للطبقة
        :param n_neurons: عدد الخلايا العصبية في هذه الطبقة
        :param weight_scale: معامل مقياس التهيئة العشوائية للأوزان (الافتراضي 0.01)
        """
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        # تهيئة غاوسية صغيرة للأوزان وانحيازات مصفوفة بأصفار
        self.weights = weight_scale * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # متغيرات لحفظ المدخلات والتدرجات
        self.inputs = None
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        التمرير الأمامي (Forward Pass)
        حساب الجداء النقطي بين مصفوفة المدخلات ومصفوفة الأوزان وإضافة متجه الإنحياز.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        التمرير العكسي (Backward Pass)
        حساب تدرجات الخسارة بالنسبة للأوزان والإنحيازات والمدخلات بتطبيق قاعدة السلسلة (Chain Rule).
        """
        # مشتقة الخسارة بالنسبة للأوزان: dL/dW = X^T · dL/dZ
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # مشتقة الخسارة بالنسبة للإنحيازات: dL/dB = sum(dL/dZ) عبر العينات
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # مشتقة الخسارة بالنسبة للمدخلات لتمريرها للطبقة السابقة: dL/dX = dL/dZ · W^T
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
`,
  },
  {
    name: 'activations.py',
    path: 'minideep_engine/activations.py',
    descriptionAr: 'دوال التفعيل غير الخطية (ReLU للتطبيقات العميقة، و Sigmoid للاحتمالات والتصنيف الثنائي) مع مشتقاتها.',
    descriptionEn: 'Nonlinear activation functions (ReLU for hidden layers, Sigmoid for bounded probabilities) and their gradients.',
    badge: 'Non-Linearity',
    exports: ['Activation_ReLU', 'Activation_Sigmoid'],
    code: `"""
activations.py - دوال التفعيل ومشتقاتها (Activation Functions & Derivatives)
منظومة MiniDeepEngine
"""

import numpy as np


class Activation_ReLU:
    """
    دالة التفعيل الخطية المصححة (Rectified Linear Unit - ReLU)
    
    المعادلة الأمامية:
        f(x) = max(0, x)
        
    المعادلة العكسية (المشتقة):
        f'(x) = 1  إذا كان x > 0
        f'(x) = 0  إذا كان x <= 0
    """
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dvalues = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # نسخ التدرجات وتصفير العناصر التي كانت مدخلاتها سالبة أو صفراً
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0
        return self.dvalues


class Activation_Sigmoid:
    """
    دالة التفعيل اللوجستية (Sigmoid Activation)
    
    المعادلة الأمامية:
        σ(z) = 1 / (1 + e^(-z))
        تحول القيم إلى احتمالات قطعية محصورة بين 0 و 1
        
    المعادلة العكسية (المشتقة):
        dσ/dz = σ(z) * (1 - σ(z))
    """
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dvalues = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        # قص المدخلات لتجنب طفح القيم العددية (Numerical Overflow) في الدالة الأسية
        clipped_inputs = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-clipped_inputs))
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # تدرج دالة Sigmoid: dL/dz = dL/dOutput * Output * (1 - Output)
        self.dvalues = dvalues * self.output * (1 - self.output)
        return self.dvalues
`,
  },
  {
    name: 'losses.py',
    path: 'minideep_engine/losses.py',
    descriptionAr: 'دوال قياس الخسارة (MSE للانحدار والتسعير، و Binary Cross-Entropy للتصنيف وتحليل السلوك).',
    descriptionEn: 'Loss criteria & analytical gradients: MSE for regression and Binary Cross-Entropy for classification.',
    badge: 'Objective Functions',
    exports: ['Loss_MSE', 'Loss_BinaryCrossEntropy'],
    code: `"""
losses.py - دوال قياس الخسارة ومشتقاتها (Loss Functions & Gradients)
منظومة MiniDeepEngine
"""

import numpy as np


class Loss_MSE:
    """
    متوسط مربع الخطأ (Mean Squared Error - MSE Loss)
    تستخدم أساساً لمهام الانحدار والتسعير والتنبؤ الرقمي المستمر (Regression Tasks).
    
    المعادلة الأمامية:
        L = (1/N) * Σ (y_pred - y_true)^2
        
    المعادلة العكسية (المشتقة بالنسبة للتوقعات):
        ∂L/∂y_pred = (2 / N) * (y_pred - y_true)
    """
    def __init__(self):
        self.dvalues = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = len(y_pred)
        self.dvalues = 2 * (y_pred - y_true) / samples
        return self.dvalues


class Loss_BinaryCrossEntropy:
    """
    دالة خسارة الإنتروبيا المتقاطعة الثنائية (Binary Cross-Entropy / Log Loss)
    تستخدم لمهام التصنيف الثنائي وتحليل السلوك التنبؤي (Binary Classification).
    
    المعادلة الأمامية:
        L = - (1/N) * Σ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]
        
    المعادلة العكسية (المشتقة بالنسبة للتوقعات):
        ∂L/∂ŷ = - (1/N) * [ (y / ŷ) - ((1 - y) / (1 - ŷ)) ]
    """
    def __init__(self):
        self.dvalues = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # قص التوقعات لتجنب حساب log(0) الذي يسبب قيمة غير معرفة (NaN)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(np.mean(sample_losses))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dvalues = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / samples
        return self.dvalues
`,
  },
  {
    name: 'optimizers.py',
    path: 'minideep_engine/optimizers.py',
    descriptionAr: 'خوارزميات تحديث الأوزان التكيفية (محسّن Adam مع تصحيح الانحياز، و SGD مع الزخم).',
    descriptionEn: 'Weight optimizers: Adaptive Moment Estimation (Adam) with bias corrections, and classical SGD with momentum.',
    badge: 'Gradient Descent',
    exports: ['Optimizer_Adam', 'Optimizer_SGD'],
    code: `"""
optimizers.py - خوارزميات تحسين وتحديث الأوزان (Weight Optimization Algorithms)
منظومة MiniDeepEngine
"""

import numpy as np


class Optimizer_Adam:
    """
    محسّن تقدير العزم التكيفي (Adaptive Moment Estimation - Adam Optimizer)
    يجمع بين ميزات Momentum (العزم الأول) و RMSprop (العزم الثاني التربيعي)
    لتحديث معدل التعلم تلقائياً لكل وزن على حدة.
    """
    def __init__(self, learning_rate: float = 0.03, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        # تهيئة ذاكرة العزوم إن لم تكن مهيأة مسبقاً في الطبقة
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        self.iterations += 1
        t = self.iterations

        # 1. تحديث عزم الأوزان (Momentum & Cache)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        
        # تصحيح الانحياز الأولي (Bias Correction)
        w_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** t)
        w_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** t)
        
        # تطبيق التحديث على الأوزان
        layer.weights -= self.learning_rate * w_momentums_corrected / (np.sqrt(w_cache_corrected) + self.epsilon)

        # 2. تحديث عزم الإنحيازات (Momentum & Cache)
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)
        
        b_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** t)
        b_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** t)
        
        # تطبيق التحديث على الإنحيازات
        layer.biases -= self.learning_rate * b_momentums_corrected / (np.sqrt(b_cache_corrected) + self.epsilon)


class Optimizer_SGD:
    """
    الانحدار التدرجي العشوائي الأساسي مع الزخم الاختياري (Stochastic Gradient Descent)
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer):
        if self.momentum > 0:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            layer.weight_momentums = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            layer.bias_momentums = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases

            layer.weights += layer.weight_momentums
            layer.biases += layer.bias_momentums
        else:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases
`,
  },
  {
    name: 'utils.py',
    path: 'minideep_engine/utils.py',
    descriptionAr: 'دوال الحفظ والاسترجاع للبيانات الخفيفة والشرائح المدمجة (JSON Serialization بدون حزم خارجية).',
    descriptionEn: 'Lightweight JSON serialization and instant layer hydration utilities for edge IoT & TinyML.',
    badge: 'IoT & Serialization',
    exports: ['save_model', 'load_model'],
    code: `"""
utils.py - دوال الحفظ والاسترجاع للبيانات الخفيفة (Model Serialization & Edge Utils)
منظومة MiniDeepEngine
"""

import json
import numpy as np
from .layers import DenseLayer


def save_model(filepath: str, *layers: DenseLayer):
    """
    حفظ أوزان وانحيازات طبقات النموذج في ملف JSON خفيف جداً
    مناسب لأنظمة إنترنت الأشياء والشرائح المدمجة (IoT / TinyML).
    
    :param filepath: مسار حفظ الملف (مثل 'minideep_model.json')
    :param layers: كائنات DenseLayer المراد حفظ معاملاتها بالترتيب
    """
    model_data = {}
    for idx, layer in enumerate(layers, start=1):
        model_data[f"layer{idx}_weights"] = layer.weights.tolist()
        model_data[f"layer{idx}_biases"] = layer.biases.tolist()
        
    model_data["__metadata__"] = {
        "engine": "MiniDeepEngine",
        "format_version": "1.0",
        "num_layers": len(layers)
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=4, ensure_ascii=False)
        
    print(f"[+] تم حفظ النموذج بنجاح في: {filepath} ({len(layers)} طبقات)")


def load_model(filepath: str, layer_shapes: list):
    """
    استرجاع أوزان النموذج من ملف JSON وبناء طبقات جديدة وحقن الأوزان فيها فورياً
    دون أي حاجة لإعادة التدريب (Zero Retraining).
    
    :param filepath: مسار ملف JSON المحفوظ
    :param layer_shapes: قائمة بأبعاد الطبقات [(n_in_1, n_out_1), (n_in_2, n_out_2), ...]
    :return: قائمة بالطبقات الجاهزة للاستنتاج الفوري (Inference)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    loaded_layers = []
    for idx, (n_in, n_out) in enumerate(layer_shapes, start=1):
        w_key = f"layer{idx}_weights"
        b_key = f"layer{idx}_biases"
        
        if w_key not in model_data or b_key not in model_data:
            raise KeyError(f"المعاملات {w_key} أو {b_key} غير موجودة في ملف النموذج {filepath}")

        # بناء طبقة جديدة تماماً
        layer = DenseLayer(n_inputs=n_in, n_neurons=n_out)
        
        # حقن الأوزان والانحيازات
        layer.weights = np.array(model_data[w_key], dtype=np.float64)
        layer.biases = np.array(model_data[b_key], dtype=np.float64)
        loaded_layers.append(layer)

    print(f"[+] تم استرجاع أوزان النموذج بنجاح من: {filepath} ({len(loaded_layers)} طبقات)")
    return loaded_layers
`,
  },
  {
    name: 'example_usage.py',
    path: 'minideep_engine/example_usage.py',
    descriptionAr: 'برنامج تطبيقي متكامل يوضح سهولة استيراد وتشغيل MiniDeepEngine في أي مشروع بايثون خارجي.',
    descriptionEn: 'End-to-end pipeline script demonstrating clean modular imports and full workflow execution.',
    badge: 'Executable Demo',
    exports: ['Main Pipeline'],
    code: `"""
example_usage.py - مثال تطبيقي شامل يوضح كيفية استدعاء واستخدام منظومة MiniDeepEngine الموحدة
"""

import numpy as np
from minideep_engine import (
    DenseLayer,
    Activation_ReLU,
    Activation_Sigmoid,
    Loss_BinaryCrossEntropy,
    Loss_MSE,
    Optimizer_Adam,
    save_model,
    load_model,
)

print("=" * 70)
print(" منظومة MiniDeepEngine الموحدة - تجربة الاستدعاء المعياري (Modular Import)")
print("=" * 70)

# ------------------------------------------------------------------------------
# 1. تطبيق التصنيف الثنائي (Binary Classification) مع الحفظ والاسترجاع
# ------------------------------------------------------------------------------
print("\\n[1] بدء تدريب نموذج التصنيف الثنائي:")

# ميزات العينات وفئاتها
X_cls = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.8, 0.9],
    [0.9, 0.8],
    [0.1, 0.9],
    [0.9, 0.2]
])
y_cls = np.array([[0], [0], [1], [1], [0], [1]])

# بناء المعمارية باستخدام الحزمة
dense1 = DenseLayer(n_inputs=2, n_neurons=8)
act1 = Activation_ReLU()
dense2 = DenseLayer(n_inputs=8, n_neurons=1)
act2 = Activation_Sigmoid()

loss_fn_cls = Loss_BinaryCrossEntropy()
opt_cls = Optimizer_Adam(learning_rate=0.03)

# التدريب
for epoch in range(1201):
    # Forward Pass
    dense1.forward(X_cls)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    preds = act2.forward(dense2.output)

    # Loss & Backward
    loss = loss_fn_cls.forward(preds, y_cls)
    loss_fn_cls.backward(preds, y_cls)
    act2.backward(loss_fn_cls.dvalues)
    dense2.backward(act2.dvalues)
    act1.backward(dense2.dinputs)
    dense1.backward(act1.dvalues)

    # Optimize
    opt_cls.update_params(dense2)
    opt_cls.update_params(dense1)

    if epoch % 400 == 0:
        accuracy = np.mean((preds > 0.5).astype(int) == y_cls) * 100
        print(f"  Epoch {epoch:4d} | BCE Loss: {loss:.5f} | Accuracy: {accuracy:.1f}%")

# حفظ النموذج كملف خفيف
save_model("minideep_classification.json", dense1, dense2)

# استرجاع النموذج في طبقات جديدة تماماً
loaded_l1, loaded_l2 = load_model("minideep_classification.json", layer_shapes=[(2, 8), (8, 1)])

# استنتاج لعينة اختبار جديدة تماماً
X_test = np.array([[0.85, 0.85]])
loaded_l1.forward(X_test)
act1.forward(loaded_l1.output)
loaded_l2.forward(act1.output)
test_prob = act2.forward(loaded_l2.output)[0][0]
test_class = 1 if test_prob > 0.5 else 0

print(f"\\n[+] نتيجة اختبار العينة المسترجعة X_test = {X_test.tolist()}:")
print(f"    الاحتمال: {test_prob:.4f} | التصنيف: فئة {test_class}")

print("\\n" + "=" * 70)
print("اكتمل تشغيل النموذج المعياري بنجاح تام وبدون أي أخطاء!")
print("=" * 70)
`,
  },
];

export const UnifiedEngineStructureLab: React.FC<UnifiedEngineStructureLabProps> = ({ lang }) => {
  const [selectedFile, setSelectedFile] = useState<EngineFile>(ENGINE_FILES[0]);
  const [copiedFile, setCopiedFile] = useState<boolean>(false);
  const [simRunning, setSimRunning] = useState<boolean>(false);
  const [simLogs, setSimLogs] = useState<string[]>([]);
  const [simProgress, setSimProgress] = useState<number>(0);

  // Copy selected file code
  const handleCopyCode = () => {
    navigator.clipboard.writeText(selectedFile.code);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  // Download selected file
  const handleDownloadFile = () => {
    const blob = new Blob([selectedFile.code], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = selectedFile.name;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Download all files as a single python script / bundle text
  const handleDownloadFullBundle = () => {
    let bundle = `# ==============================================================================
# MiniDeepEngine - Unified Modular Framework Bundle
# Compiled Package Distribution (All Modules Combined)
# ==============================================================================
`;
    ENGINE_FILES.forEach((f) => {
      bundle += `\n\n# ==============================================================================\n# FILE: ${f.path}\n# ==============================================================================\n`;
      bundle += f.code;
    });

    const blob = new Blob([bundle], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'minideep_engine_complete_bundle.py';
    link.click();
    URL.revokeObjectURL(url);
  };

  // Interactive Live Pipeline Simulator
  const handleRunSim = () => {
    if (simRunning) return;
    setSimRunning(true);
    setSimLogs([]);
    setSimProgress(0);

    const logs: string[] = [
      `>>> from minideep_engine import DenseLayer, Activation_ReLU, Activation_Sigmoid`,
      `>>> from minideep_engine import Loss_BinaryCrossEntropy, Optimizer_Adam, save_model, load_model`,
      `[i] تم استيراد حزمة MiniDeepEngine بنجاح (Zero External Dependencies - Pure NumPy)`,
      `----------------------------------------------------------------------`,
      `[1/4] بناء معمارية الشبكة: Dense(2, 8) -> ReLU -> Dense(8, 1) -> Sigmoid`,
      `[2/4] تجهيز ميزات التصنيف الثنائي (6 عينات)...`,
    ];

    let step = 0;
    const interval = setInterval(() => {
      step++;
      setSimProgress(step * 25);

      if (step === 1) {
        logs.push(`Epoch    0 | Loss: 0.69315 | Accuracy: 50.0% | Grad Norm: OK`);
        setSimLogs([...logs]);
      } else if (step === 2) {
        logs.push(`Epoch  400 | Loss: 0.38412 | Accuracy: 83.3% | Adam Momentum active`);
        setSimLogs([...logs]);
      } else if (step === 3) {
        logs.push(`Epoch  800 | Loss: 0.11204 | Accuracy: 100.0% | Zero classification errors`);
        logs.push(`Epoch 1200 | Loss: 0.04189 | Accuracy: 100.0% | Full Convergence!`);
        logs.push(`[3/4] استدعاء save_model('minideep_classification.json', dense1, dense2)`);
        logs.push(`[+] تم حفظ أوزان النموذج بنجاح في minideep_classification.json (~620 bytes)`);
        setSimLogs([...logs]);
      } else if (step === 4) {
        logs.push(`[4/4] استدعاء load_model('minideep_classification.json', [(2,8), (8,1)])`);
        logs.push(`[+] تم استرجاع النموذج في كائنات جديدة وحقن الأوزان بدون إعادة تدريب!`);
        logs.push(`>>> اختبار عينة جديدة: X_test = [[0.85, 0.85]]`);
        logs.push(`>>> الاحتمال المتوقع (Sigmoid Output): 0.9684 (96.8%)`);
        logs.push(`>>> القرار النهائي: فئة 1 (Class 1 - سلوك مرتفع) ✓`);
        logs.push(`======================================================================`);
        logs.push(`[✓] اكتملت دورة العمل بالكامل بنجاح تام وفق الهيكل الموحد لمنظومة MiniDeepEngine!`);
        setSimLogs([...logs]);
        setSimRunning(false);
        clearInterval(interval);
      }
    }, 450);
  };

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/70 to-slate-900 border border-indigo-500/30 rounded-2xl p-5 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 relative z-10">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="p-2 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                <FolderTree className="w-5 h-5" />
              </span>
              <h2 className="text-xl sm:text-2xl font-black text-white tracking-wide">
                {lang === 'ar' ? 'الهيكل الموحد لمنظومة MiniDeepEngine' : 'MiniDeepEngine Unified Modular Package'}
              </h2>
              <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono">
                v1.0.0 Modular
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-300 max-w-3xl leading-relaxed">
              {lang === 'ar'
                ? 'تجميع كافة المكونات الرياضية والطبقات والمحسّنات وأدوات الحفظ في حزمة بايثون معيارية منظمة جاهزة للاستيراد المباشر في أي مشروع خارجي.'
                : 'Packaging all mathematical layers, activations, losses, optimizers, and serialization utils into a production-ready Python package.'}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadFullBundle}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/25 transition"
            >
              <Download className="w-4 h-4" />
              <span>{lang === 'ar' ? 'تنزيل الحزمة كاملة (Full Bundle)' : 'Download Full Package'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid: File Tree (Left) vs Code Inspector (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Interactive Package Tree (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Tree Explorer Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
                <Package className="w-4 h-4 text-indigo-400" />
                <span>minideep_engine/</span>
              </div>
              <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                {ENGINE_FILES.length} files
              </span>
            </div>

            {/* File List Items */}
            <div className="space-y-1.5">
              {ENGINE_FILES.map((file) => {
                const isSelected = selectedFile.name === file.name;
                return (
                  <button
                    key={file.name}
                    onClick={() => setSelectedFile(file)}
                    className={`w-full text-right rtl:text-right ltr:text-left p-2.5 rounded-xl transition flex items-center justify-between border ${
                      isSelected
                        ? 'bg-indigo-600/20 border-indigo-500/50 text-white shadow-md'
                        : 'bg-slate-950/60 border-slate-800/80 text-slate-400 hover:text-white hover:bg-slate-800/60'
                    }`}
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <FileCode className={`w-4 h-4 shrink-0 ${isSelected ? 'text-indigo-400' : 'text-slate-500'}`} />
                      <div className="truncate">
                        <div className="text-xs font-mono font-bold">{file.name}</div>
                        <div className="text-[10px] text-slate-500 truncate">
                          {lang === 'ar' ? file.descriptionAr : file.descriptionEn}
                        </div>
                      </div>
                    </div>

                    <span className="text-[9px] font-mono shrink-0 px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                      {file.badge}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Clean Architecture Advantages */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-800 text-xs font-bold text-slate-200">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>{lang === 'ar' ? 'مزايا الهيكل المعياري (Architecture)' : 'Modular Design Benefits'}</span>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <div className="p-2 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-0.5">
                <div className="font-bold text-sky-300 text-[11px]">1. الاستدعاء بنقرة واحدة (Single Import)</div>
                <p className="text-slate-400 text-[10px] font-mono">
                  from minideep_engine import DenseLayer, Activation_ReLU...
                </p>
              </div>

              <div className="p-2 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-0.5">
                <div className="font-bold text-emerald-300 text-[11px]">2. فصل المهام والمسؤوليات (Decoupled)</div>
                <p className="text-slate-400 text-[10px]">
                  عزل طبقات الحساب عن دوال التفعيل والمحسّنات يتيح التعديل والتوسع بسهولة تامة.
                </p>
              </div>

              <div className="p-2 rounded-lg bg-slate-950/70 border border-slate-800/80 space-y-0.5">
                <div className="font-bold text-amber-300 text-[11px]">3. خفيف ومثالي لـ IoT و MicroPython</div>
                <p className="text-slate-400 text-[10px]">
                  لا توجد أي حزم خارجية سوى NumPy، مع قابلية النقل الفوري لأجهزة الراسبيري باي والـ ESP32.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Code Inspector & Live Runner (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {/* File Header Bar & Actions */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm sm:text-base font-bold text-white flex items-center gap-1.5">
                    <FileCode className="w-4 h-4 text-indigo-400" />
                    {selectedFile.path}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    {selectedFile.badge}
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  {lang === 'ar' ? selectedFile.descriptionAr : selectedFile.descriptionEn}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyCode}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition"
                >
                  {copiedFile ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedFile ? (lang === 'ar' ? 'تم النسخ' : 'Copied') : (lang === 'ar' ? 'نسخ الكود' : 'Copy')}</span>
                </button>

                <button
                  onClick={handleDownloadFile}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>{lang === 'ar' ? 'تنزيل الملف' : 'Download'}</span>
                </button>
              </div>
            </div>

            {/* Exported Symbols Chips */}
            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
              <span className="text-slate-500">{lang === 'ar' ? 'العناصر المصدرة (__all__):' : 'Exports:'}</span>
              {selectedFile.exports.map((exp) => (
                <span
                  key={exp}
                  className="font-mono text-[10px] px-2 py-0.5 rounded-md bg-slate-950 border border-slate-800 text-indigo-300 font-semibold"
                >
                  {exp}
                </span>
              ))}
            </div>

            {/* Code Pre Block */}
            <div className="relative">
              <pre className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-slate-300 overflow-x-auto max-h-[420px] border border-slate-800/80 leading-relaxed">
                <code>{selectedFile.code}</code>
              </pre>
            </div>
          </div>

          {/* Interactive Live Modular Pipeline Runner */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-emerald-400" />
                <h4 className="font-bold text-white text-sm">
                  {lang === 'ar' ? 'محاكي التشغيل والاستدعاء المباشر (Modular Pipeline Demo)' : 'Modular Pipeline Live Runner'}
                </h4>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleRunSim}
                  disabled={simRunning}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold shadow-lg shadow-emerald-600/25 transition"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{simRunning ? (lang === 'ar' ? 'جاري التشغيل...' : 'Running...') : (lang === 'ar' ? 'تشغيل النموذج المعياري' : 'Run Pipeline')}</span>
                </button>
              </div>
            </div>

            {/* Progress Bar */}
            {simRunning && (
              <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="bg-emerald-500 h-full transition-all duration-300"
                  style={{ width: `${simProgress}%` }}
                />
              </div>
            )}

            {/* Console Terminal Output */}
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-[11px] text-slate-300 min-h-36 max-h-56 overflow-y-auto space-y-1">
              {simLogs.length === 0 ? (
                <div className="text-slate-600 italic text-center py-8">
                  {lang === 'ar'
                    ? 'اضغط على "تشغيل النموذج المعياري" لمحاكاة تنفيذ استدعاء حزمة MiniDeepEngine وحفظ واسترجاع النموذج لحظياً'
                    : 'Click "Run Pipeline" to simulate importing minideep_engine and running full training & serialization.'}
                </div>
              ) : (
                simLogs.map((log, idx) => (
                  <div
                    key={idx}
                    className={
                      log.startsWith('>>>')
                        ? 'text-sky-300 font-bold'
                        : log.startsWith('[+]') || log.startsWith('[✓]')
                        ? 'text-emerald-400 font-semibold'
                        : log.startsWith('Epoch')
                        ? 'text-amber-300'
                        : 'text-slate-400'
                    }
                  >
                    {log}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
