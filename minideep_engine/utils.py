"""
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
