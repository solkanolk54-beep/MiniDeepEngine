"""
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
