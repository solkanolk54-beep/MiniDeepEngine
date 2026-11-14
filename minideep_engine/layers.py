"""
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
