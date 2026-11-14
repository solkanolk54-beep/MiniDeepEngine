"""
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
