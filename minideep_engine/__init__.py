"""
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
