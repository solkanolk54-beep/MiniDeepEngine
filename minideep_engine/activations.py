"""
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
