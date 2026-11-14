"""
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
print("\n[1] بدء تدريب نموذج التصنيف الثنائي:")

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

print(f"\n[+] نتيجة اختبار العينة المسترجعة X_test = {X_test.tolist()}:")
print(f"    الاحتمال: {test_prob:.4f} | التصنيف: فئة {test_class}")

print("\n" + "=" * 70)
print("اكتمل تشغيل النموذج المعياري بنجاح تام وبدون أي أخطاء!")
print("=" * 70)
