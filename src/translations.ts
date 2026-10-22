export type Language = 'ar' | 'en';

export interface Translations {
  appTitle: string;
  appSubtitle: string;
  playgroundTab: string;
  mathInspectorTab: string;
  architectureTab: string;
  codeExportTab: string;
  mathGuideTab: string;
  
  // Controls
  train: string;
  pause: string;
  step: string;
  reset: string;
  speed: string;
  epochs: string;
  loss: string;
  accuracy: string;
  rmse: string;
  learningRate: string;
  selectDataset: string;
  
  // Visuals
  decisionBoundary: string;
  lossCurve: string;
  networkArchitecture: string;
  clickToInspect: string;
  clickToAddPoint: string;
  positiveClass: string;
  negativeClass: string;
  
  // Math Step
  step1Title: string;
  step1Desc: string;
  step2Title: string;
  step2Desc: string;
  step3Title: string;
  step3Desc: string;
  step4Title: string;
  step4Desc: string;
  step5Title: string;
  step5Desc: string;
  nextStep: string;
  prevStep: string;
  autoPlaySteps: string;
  
  // Builder
  layersTitle: string;
  addDense: string;
  addActivation: string;
  addDropout: string;
  optimizerSettings: string;
  lossSettings: string;
  neurons: string;
  activation: string;
  dropoutRate: string;
  removeLayer: string;
  applyArchitecture: string;
  presetsTitle: string;
  downloadPython: string;
  copyCode: string;
  copied: string;
  exportJSON: string;
  importJSON: string;
  
  // R&D & Math Logger Lab
  rdLabTab: string;
  mathLoggerTitle: string;
  customActivationTitle: string;
  liveTerminalLogs: string;
  clearLogs: string;
  copyLogs: string;
  testCustomActivation: string;
  injectActivation: string;
  activationInjected: string;
  curveComparison: string;

  // XOR Benchmark
  xorBenchmarkTab: string;
  xorSubtitle: string;
  xorLinearPerceptron: string;
  xorMultiLayer: string;
  xorTruthTable: string;
  xorDecisionBoundary: string;
  xorRun1500: string;
  xorLossCurve: string;
  xorWeightsChainRule: string;
  xorPythonCode: string;
}

export const translations: Record<Language, Translations> = {
  ar: {
    appTitle: 'منظومة MiniDeepEngine',
    appSubtitle: 'نواة التعلم العميق والشبكات العصبية المبنية من الصفر خطوة بخطوة',
    playgroundTab: 'المختبر والتصور الحي',
    mathInspectorTab: 'التشريح الرياضي المتسلسل',
    architectureTab: 'بناء معمارية الشبكة',
    codeExportTab: 'الكود وتصدير بايثون',
    mathGuideTab: 'الدليل الرياضي الشامل',
    
    train: 'بدء التدريب',
    pause: 'إيقاف مؤقت',
    step: 'خطوة واحدة (Epoch)',
    reset: 'إعادة ضبط الأوزان',
    speed: 'سرعة التدريب',
    epochs: 'الدورات (Epochs)',
    loss: 'قيمة الخسارة (Loss)',
    accuracy: 'دقة النموذج (Accuracy)',
    rmse: 'جذر متوسط مربع الخطأ (RMSE)',
    learningRate: 'معدل التعلم (LR)',
    selectDataset: 'اختيار مجموعة البيانات',
    
    decisionBoundary: 'خريطة فضاء القرار (Decision Boundary)',
    lossCurve: 'منحنى الخسارة والأداء (Loss & Metric Curve)',
    networkArchitecture: 'الهيكل العصبي وتدفق الأوزان (Computation Graph)',
    clickToInspect: 'اضغط على أي عصبون أو وزن لتشريح بياناته وحساباته التفصيلية',
    clickToAddPoint: 'انقر على الخريطة لإضافة نقاط جديدة وتجربة قدرة الشبكة على الفصل',
    positiveClass: 'الفئة الموجبة (1)',
    negativeClass: 'الفئة السالبة (0)',
    
    step1Title: 'المرحلة 1: الممر الأمامي (Forward Pass)',
    step1Desc: 'ضرب مصفوفة المدخلات في الأوزان وإضافة الانحياز: z = W • a + b',
    step2Title: 'المرحلة 2: دوال التفعيل (Activation Functions)',
    step2Desc: 'تطبيق اللاخطية على المخرجات: a = σ(z) مثل ReLU و Sigmoid و Tanh',
    step3Title: 'المرحلة 3: حساب دالة الخسارة (Loss Function)',
    step3Desc: 'قياس مقدار الخطأ بين التوقع والقيمة المستهدفة باستخدام MSE أو Cross-Entropy',
    step4Title: 'المرحلة 4: الانتشار الخلفي وقاعدة السلسلة (Backpropagation)',
    step4Desc: 'حساب التدرجات (Gradients) لكل وزن وانحياز ومدخل عبر Chain Rule',
    step5Title: 'المرحلة 5: تحديث الأوزان بالمحسّن آدم (Adam Optimizer)',
    step5Desc: 'تطبيق العزم الأول m_t والثاني v_t مع تصحيح الانحياز وتحديث الأوزان',
    nextStep: 'الخطوة التالية ◀',
    prevStep: '▶ الخطوة السابقة',
    autoPlaySteps: 'تشغيل تسلسلي',
    
    layersTitle: 'طبقات الشبكة العصبية',
    addDense: '+ إضافة طبقة متصلة (Dense Layer)',
    addActivation: '+ إضافة دالة تفعيل (Activation)',
    addDropout: '+ إضافة طبقة إسقاط عشوائي (Dropout)',
    optimizerSettings: 'إعدادات المحسّن (Optimizer)',
    lossSettings: 'دالة الخسارة (Loss Function)',
    neurons: 'عدد العصبونات',
    activation: 'دالة التفعيل',
    dropoutRate: 'نسبة الإسقاط',
    removeLayer: 'حذف الطبقة',
    applyArchitecture: 'تطبيق وإعادة بناء النموذج',
    
    downloadPython: 'تحميل ملف بايثون (minideepengine.py)',
    copyCode: 'نسخ الكود الكامل',
    copied: 'تم النسخ بنجاح!',
    exportJSON: 'تصدير النموذج (JSON)',
    importJSON: 'استيراد نموذج جاهز',
    presetsTitle: 'نماذج جاهزة سريعة',

    rdLabTab: 'مختبر الأبحاث وسجل التشريح (R&D Lab)',
    mathLoggerTitle: 'سِجِل التشريح الرياضي الحي (Math Inspector Logger)',
    customActivationTitle: 'مختبر دوال التفعيل المخصصة (Custom Activation Sandbox)',
    liveTerminalLogs: 'سجل التدرجات والعمليات لحظياً',
    clearLogs: 'مسح السجلات',
    copyLogs: 'نسخ مخرجات السجل',
    testCustomActivation: 'اختبار الدالة وحساب المشتقة',
    injectActivation: 'حقن الدالة في الطبقات المخفية للنموذج',
    activationInjected: 'تم حقن الدالة المخصصة بنجاح في معمارية النموذج!',
    curveComparison: 'مقارنة دالة التفعيل مع مشتقتها f(x) vs f\'(x)',

    // XOR Benchmark
    xorBenchmarkTab: 'معضلة XOR والتصنيف غير الخطي (XOR Benchmark)',
    xorSubtitle: 'اختبار المعيار الكلاسيكي للشبكات العصبية: كيف تحل الطبقة المخفية مع دالة ReLU ما عجزت عنه الخطوط المستقيمة',
    xorLinearPerceptron: 'البرسبترون الخطي البسيط (بدون طبقة مخفية - سيعجز عن حل XOR)',
    xorMultiLayer: 'شبكة متعددة الطبقات (طبقة مخفية 4 خلايا + ReLU - ستحل XOR بنسبة 100%)',
    xorTruthTable: 'جدول الحقيقة والتنبؤات اللحظية (Truth Table)',
    xorDecisionBoundary: 'خريطة حدود القرار الثنائية (Decision Boundary)',
    xorRun1500: 'تشغيل 1500 دورة تدريبية',
    xorLossCurve: 'منحنى الخسارة (MSE Loss Curve)',
    xorWeightsChainRule: 'تشريح الأوزان وتدرجات قاعدة السلسلة (Chain Rule)',
    xorPythonCode: 'كود بايثون المستقل (NumPy + Matplotlib)',
  },
  en: {
    appTitle: 'MiniDeepEngine Studio',
    appSubtitle: 'Deep Learning & Neural Network Framework built from scratch with pure math',
    playgroundTab: 'Playground & Live Training',
    mathInspectorTab: 'Mathematical Step Inspector',
    architectureTab: 'Architecture Builder',
    codeExportTab: 'Python Code & Export',
    mathGuideTab: 'Mathematical Guide',
    
    train: 'Train Model',
    pause: 'Pause',
    step: 'Step (1 Epoch)',
    reset: 'Reset Weights',
    speed: 'Training Speed',
    epochs: 'Epochs',
    loss: 'Loss Value',
    accuracy: 'Accuracy',
    rmse: 'RMSE',
    learningRate: 'Learning Rate',
    selectDataset: 'Select Dataset',
    
    decisionBoundary: 'Decision Boundary Surface',
    lossCurve: 'Loss & Metric Progression Curve',
    networkArchitecture: 'Neural Computation Graph',
    clickToInspect: 'Click on any neuron or weight synapse to inspect gradients and math',
    clickToAddPoint: 'Click on the canvas to add custom data points',
    positiveClass: 'Class 1 (Positive)',
    negativeClass: 'Class 0 (Negative)',
    
    step1Title: 'Stage 1: Forward Pass',
    step1Desc: 'Linear transformation: z = W • a + b',
    step2Title: 'Stage 2: Activation Function',
    step2Desc: 'Non-linear mapping: a = σ(z) (ReLU, Sigmoid, Tanh, LeakyReLU)',
    step3Title: 'Stage 3: Loss Computation',
    step3Desc: 'Measure discrepancy between prediction and ground truth (MSE, Cross-Entropy)',
    step4Title: 'Stage 4: Backpropagation & Chain Rule',
    step4Desc: 'Compute gradients for weights (dW), biases (db), and inputs via Chain Rule',
    step5Title: 'Stage 5: Parameter Update (Adam Optimizer)',
    step5Desc: 'First moment m_t, second moment v_t, bias correction and parameter update',
    nextStep: 'Next Step ▶',
    prevStep: '◀ Prev Step',
    autoPlaySteps: 'Auto Play',
    
    layersTitle: 'Neural Network Layers',
    addDense: '+ Add Dense Layer',
    addActivation: '+ Add Activation',
    addDropout: '+ Add Dropout Layer',
    optimizerSettings: 'Optimizer Configuration',
    lossSettings: 'Loss Function',
    neurons: 'Neurons Count',
    activation: 'Activation Function',
    dropoutRate: 'Dropout Rate',
    removeLayer: 'Remove Layer',
    applyArchitecture: 'Build & Initialize Model',
    
    downloadPython: 'Download Python File (minideepengine.py)',
    copyCode: 'Copy Complete Code',
    copied: 'Copied to Clipboard!',
    exportJSON: 'Export Model (JSON)',
    importJSON: 'Import Model (JSON)',
    presetsTitle: 'Model Architecture Presets',

    rdLabTab: 'R&D Lab & Math Logger',
    mathLoggerTitle: 'Live Mathematical Inspector Logger',
    customActivationTitle: 'Custom Activation Function Sandbox',
    liveTerminalLogs: 'Live Gradient & Layer Logs',
    clearLogs: 'Clear Logs',
    copyLogs: 'Copy Terminal Output',
    testCustomActivation: 'Test Function & Derivative',
    injectActivation: 'Inject into Model Hidden Layers',
    activationInjected: 'Custom activation successfully injected into model!',
    curveComparison: 'Function & Derivative Curve Comparison: f(x) vs f\'(x)',

    // XOR Benchmark
    xorBenchmarkTab: 'XOR Non-Linearity Benchmark',
    xorSubtitle: 'Classic benchmark proving why hidden layers and non-linear ReLU overcome linear boundaries',
    xorLinearPerceptron: 'Linear Perceptron (No hidden layer - cannot solve XOR)',
    xorMultiLayer: '2-Layer MLP (4 hidden neurons + ReLU - solves XOR to 100%)',
    xorTruthTable: 'Truth Table & Real-Time Predictions',
    xorDecisionBoundary: 'Decision Boundary & Probability Map',
    xorRun1500: 'Run Full 1500 Epochs',
    xorLossCurve: 'MSE Loss Progression Curve',
    xorWeightsChainRule: 'Weights & Chain Rule Gradient Inspector',
    xorPythonCode: 'Standalone Python Code (NumPy + Matplotlib)',
  },
};
