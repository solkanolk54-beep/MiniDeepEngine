import React, { useState } from 'react';
import { ChevronRight, ChevronLeft, Play, ArrowRight, BookOpen, Check, Layers, Cpu, Compass } from 'lucide-react';
import { NeuralNetwork } from '../engine/model';
import { DenseLayer } from '../engine/layers';
import { Language, translations } from '../translations';
import { MatrixOps } from '../engine/matrix';

interface MathStepInspectorProps {
  model: NeuralNetwork;
  lang: Language;
  onExecuteSingleStep: () => void;
}

function formatMatrixDisplay(m?: number[][], maxRows = 3, maxCols = 4): string {
  if (!m || m.length === 0) return '[\n  [ 0.000, 0.000 ]\n]';
  const rows = m.slice(0, maxRows).map(row => {
    const formattedCols = row.slice(0, maxCols).map(v => Number(v.toFixed(3)).toString().padStart(6, ' ')).join(', ');
    return `  [ ${formattedCols} ]`;
  }).join('\n');
  return `[\n${rows}\n]`;
}

export const MathStepInspector: React.FC<MathStepInspectorProps> = ({
  model,
  lang,
  onExecuteSingleStep,
}) => {
  const [activeStep, setActiveStep] = useState<number>(1);
  const t = translations[lang];

  const denseLayers = model.layers.filter(l => l instanceof DenseLayer) as DenseLayer[];
  const sampleLayer = denseLayers[0];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Title & Stage Navigator */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Compass className="w-5 h-5 text-indigo-400" />
            <span>{t.mathInspectorTab}</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            {lang === 'ar'
              ? 'تتبع المعادلات الرياضية الخمس للنواة البرمجية الحية كما تم شرحها وبناؤها من الصفر'
              : 'Step through the 5 foundational mathematical stages computed from scratch in real-time'}
          </p>
        </div>

        {/* Step Tabs 1-5 */}
        <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
          {[1, 2, 3, 4, 5].map((stepNum) => (
            <button
              key={stepNum}
              onClick={() => setActiveStep(stepNum)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1 ${
                activeStep === stepNum
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>{lang === 'ar' ? `المرحلة ${stepNum}` : `Stage ${stepNum}`}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Stage 1: Forward Pass */}
      {activeStep === 1 && (
        <div className="space-y-4">
          <div className="flex items-start gap-3 bg-indigo-950/30 border border-indigo-900/50 rounded-xl p-4">
            <div className="p-2 rounded-lg bg-indigo-600/20 text-indigo-400 font-bold font-mono text-sm" dir="ltr">
              z = W • a + b
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">{t.step1Title}</h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                {lang === 'ar'
                  ? 'تقوم كل طبقة متصلة (Dense Layer) بعملية تحويل خطي، حيث تُضرب مصفوفة المدخلات السابقة a في مصفوفة الأوزان W وتضاف إليها متجهات الانحياز b.'
                  : 'The Dense layer applies an affine linear transformation: the input activations vector (a) is multiplied by synaptic weight matrix (W) plus bias vector (b).'}
              </p>
            </div>
          </div>

          {/* Matrix Math Visualization */}
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400 uppercase font-bold block mb-2">
                1. {lang === 'ar' ? 'مصفوفة المدخلات X' : 'Inputs Matrix X'}
              </span>
              <pre className="text-xs font-mono text-cyan-300 bg-slate-900/80 p-3 rounded-lg overflow-x-auto text-left" dir="ltr">
                {formatMatrixDisplay(sampleLayer?.inputs)}
              </pre>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400 uppercase font-bold block mb-2">
                2. {lang === 'ar' ? 'مصفوفة الأوزان W' : 'Weights Matrix W'}
              </span>
              <pre className="text-xs font-mono text-amber-300 bg-slate-900/80 p-3 rounded-lg overflow-x-auto text-left" dir="ltr">
                {formatMatrixDisplay(sampleLayer?.weights)}
              </pre>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400 uppercase font-bold block mb-2">
                3. {lang === 'ar' ? 'المخرج الخطي z' : 'Linear Output z'}
              </span>
              <pre className="text-xs font-mono text-emerald-300 bg-slate-900/80 p-3 rounded-lg overflow-x-auto text-left" dir="ltr">
                {formatMatrixDisplay(sampleLayer?.z)}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Stage 2: Activation Functions */}
      {activeStep === 2 && (
        <div className="space-y-4">
          <div className="flex items-start gap-3 bg-violet-950/30 border border-violet-900/50 rounded-xl p-4">
            <div className="p-2 rounded-lg bg-violet-600/20 text-violet-400 font-bold font-mono text-sm">
              a = σ(z)
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">{t.step2Title}</h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                {lang === 'ar'
                  ? 'دوال التفعيل تكسر الخطية (Non-linearity) لتسمح للشبكة بحل مسائل غير قابلة للفصل خطياً مثل XOR. لكل دالة مشتقة رياضية (Derivative) خاصة بها تُستخدم أثناء الانتشار الخلفي.'
                  : 'Activation functions introduce non-linearity, enabling deep networks to solve complex decision boundaries like XOR. Each activation possesses an analytical derivative for backprop.'}
              </p>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* ReLU */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-xs">ReLU</span>
                <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-indigo-300 font-mono">f(x)=max(0,x)</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-tight">
                {lang === 'ar' ? 'المشتقة: 1 إذا x > 0 وإلا 0. سريعة الحساب وتمنع تلاشي التدرج.' : "Derivative: 1 if x > 0 else 0. Solves vanishing gradients."}
              </p>
              <div className="font-mono text-xs text-cyan-400 bg-slate-900/90 p-2 rounded">
                f'(x) = [x &gt; 0] ? 1 : 0
              </div>
            </div>

            {/* Sigmoid */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-xs">Sigmoid</span>
                <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-emerald-300 font-mono">1/(1+e⁻ˣ)</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-tight">
                {lang === 'ar' ? 'تضغط القيم بين 0 و 1 (مثالية للاحتمالات الثنائية).' : "Squashes output between 0 and 1 for binary probabilities."}
              </p>
              <div className="font-mono text-xs text-emerald-400 bg-slate-900/90 p-2 rounded">
                f'(x) = σ(x) · (1 - σ(x))
              </div>
            </div>

            {/* Tanh */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-xs">Tanh</span>
                <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-amber-300 font-mono">tanh(x)</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-tight">
                {lang === 'ar' ? 'تضغط القيم بين -1 و +1 مع مركزية الصفر.' : "Zero-centered between -1 and +1."}
              </p>
              <div className="font-mono text-xs text-amber-400 bg-slate-900/90 p-2 rounded">
                f'(x) = 1 - tanh²(x)
              </div>
            </div>

            {/* LeakyReLU */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-xs">Leaky ReLU</span>
                <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-rose-300 font-mono">max(αx, x)</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-tight">
                {lang === 'ar' ? 'تسمح بتدرج صغير عندما x < 0 لمنع موت العصبونات (Dying ReLU).' : "Prevents dying neurons with small leak gradient."}
              </p>
              <div className="font-mono text-xs text-rose-400 bg-slate-900/90 p-2 rounded">
                f'(x) = [x &gt; 0] ? 1 : α
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Stage 3: Loss Function */}
      {activeStep === 3 && (
        <div className="space-y-4">
          <div className="flex items-start gap-3 bg-cyan-950/30 border border-cyan-900/50 rounded-xl p-4">
            <div className="p-2 rounded-lg bg-cyan-600/20 text-cyan-400 font-bold font-mono text-sm">
              L = Loss(ŷ, y)
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">{t.step3Title}</h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                {lang === 'ar'
                  ? 'دالة الخسارة تقيس الفارق بين توقع الشبكة العصبية ŷ والقيمة الحقيقية y. مشتقة الخسارة بالنسبة للتوقعات هي نقطة الانطلاق في عملية الانتشار الخلفي.'
                  : 'Loss measures the error between network prediction and true targets. Its derivative is the root signal for the backward pass.'}
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
              <span className="text-xs font-bold text-cyan-400 block">Mean Squared Error (MSE)</span>
              <div className="font-mono text-xs bg-slate-900 p-2.5 rounded text-slate-300 text-left" dir="ltr">
                MSE = (1/N) · Σ (ŷᵢ - yᵢ)²
              </div>
              <div className="font-mono text-xs bg-slate-900 p-2.5 rounded text-cyan-300 text-left" dir="ltr">
                ∂L/∂ŷ = (2/N) · (ŷᵢ - yᵢ)
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {lang === 'ar' ? 'المعيار القياسي لمشاكل التنبؤ العددي والانحدار (Regression).' : 'Standard metric for continuous numeric regression.'}
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
              <span className="text-xs font-bold text-emerald-400 block">Binary Cross-Entropy (BCE)</span>
              <div className="font-mono text-xs bg-slate-900 p-2.5 rounded text-slate-300 text-left" dir="ltr">
                BCE = -(1/N) · Σ [y ln(p) + (1-y) ln(1-p)]
              </div>
              <div className="font-mono text-xs bg-slate-900 p-2.5 rounded text-emerald-300 text-left" dir="ltr">
                ∂L/∂p = - [y/p - (1-y)/(1-p)] / N
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {lang === 'ar' ? 'المعيار المثالي لمشاكل التصنيف الثنائي وتقدير الاحتمال.' : 'Optimal loss for binary probability classification.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stage 4: Backpropagation & Chain Rule */}
      {activeStep === 4 && (
        <div className="space-y-4">
          <div className="flex items-start gap-3 bg-amber-950/30 border border-amber-900/50 rounded-xl p-4">
            <div className="p-2 rounded-lg bg-amber-600/20 text-amber-400 font-bold font-mono text-sm">
              Chain Rule
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">{t.step4Title}</h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                {lang === 'ar'
                  ? 'الانتشار الخلفي يطبق قاعدة السلسلة (Chain Rule) لتوزيع إشارة الخطأ من المخرجات عائداً إلى الطبقات الأولى، لحساب كم يؤثر كل وزن وانحياز في النتيجة النهائية.'
                  : 'Backpropagation applies the Chain Rule backwards from output to input, computing how much every single weight and bias influenced total error.'}
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-3 font-mono text-xs">
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-amber-400 font-bold">1. {lang === 'ar' ? 'تدرج الأوزان (dW)' : 'Weight Gradient (dW)'}</span>
              <p className="text-slate-300 bg-slate-900 p-2 rounded text-left" dir="ltr">dW = inputsᵀ • dvalues</p>
              <p className="text-[11px] text-slate-400">
                {lang === 'ar' ? 'ضرب منقول مصفوفة المدخلات السابقة في تدرجات الطبقة.' : 'Transpose of input activations multiplied by upstream gradients.'}
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-emerald-400 font-bold">2. {lang === 'ar' ? 'تدرج الانحياز (db)' : 'Bias Gradient (db)'}</span>
              <p className="text-slate-300 bg-slate-900 p-2 rounded text-left" dir="ltr">db = Σ dvalues (axis=0)</p>
              <p className="text-[11px] text-slate-400">
                {lang === 'ar' ? 'مجموع التدرجات على طول عينات الدفعة.' : 'Sum of gradients across the batch dimension.'}
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-1.5">
              <span className="text-cyan-400 font-bold">3. {lang === 'ar' ? 'تدرج المدخلات (dinputs)' : 'Input Gradient (dX)'}</span>
              <p className="text-slate-300 bg-slate-900 p-2 rounded text-left" dir="ltr">dX = dvalues • weightsᵀ</p>
              <p className="text-[11px] text-slate-400">
                {lang === 'ar' ? 'تمرير التدرج للطبقة السابقة لمتابعة السلسلة.' : 'Passes gradients back to preceding layer in the network.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stage 5: Adam Optimizer */}
      {activeStep === 5 && (
        <div className="space-y-4">
          <div className="flex items-start gap-3 bg-emerald-950/30 border border-emerald-900/50 rounded-xl p-4">
            <div className="p-2 rounded-lg bg-emerald-600/20 text-emerald-400 font-bold font-mono text-sm" dir="ltr">
              Adam Update
            </div>
            <div>
              <h3 className="font-bold text-white text-sm">{t.step5Title}</h3>
              <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                {lang === 'ar'
                  ? 'خوارزمية Adam (Adaptive Moment Estimation) تجمع بين العزم الأول m_t (Momentum) والعزم الثاني v_t (RMSprop) مع تصحيح الانحياز (Bias Correction) لتحديث الأوزان بكفاءة وثبات فائقين.'
                  : 'Adam integrates first moment momentum and second moment gradient variance with initial bias correction for adaptive parameter tuning.'}
              </p>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-4 font-mono text-xs">
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
              <span className="text-cyan-400 font-bold block">1. {lang === 'ar' ? 'العزم الأول والثاني (Moments)' : 'First & Second Moments'}</span>
              <div className="bg-slate-900 p-2 rounded text-slate-200 text-left" dir="ltr">
                mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ
              </div>
              <div className="bg-slate-900 p-2 rounded text-slate-200 text-left" dir="ltr">
                vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²
              </div>
              <p className="text-[11px] text-slate-400">
                {lang === 'ar' ? 'تخزين المتوسط المتحرك للتدرجات ومربعاتها (β₁=0.9, β₂=0.999).' : 'Moving average of gradients and squared gradients.'}
              </p>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
              <span className="text-emerald-400 font-bold block">2. {lang === 'ar' ? 'تصحيح الانحياز والتحديث النهائي' : 'Bias Correction & Final Update'}</span>
              <div className="bg-slate-900 p-2 rounded text-emerald-300 text-left" dir="ltr">
                m̂ₜ = mₜ / (1 - β₁ᵗ) , v̂ₜ = vₜ / (1 - β₂ᵗ)
              </div>
              <div className="bg-slate-900 p-2 rounded text-cyan-300 text-left" dir="ltr">
                W ← W - (η / (√v̂ₜ + ε)) · m̂ₜ
              </div>
              <p className="text-[11px] text-slate-400">
                {lang === 'ar' ? 'التصحيح يمنع اقتراب القيم من الصفر في البدايات وتحديد الخطوة المثلى.' : 'Corrects zero-bias in initial epochs and steps smoothly.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Step Navigation Controls */}
      <div className="flex items-center justify-between pt-4 border-t border-slate-800">
        <button
          onClick={() => setActiveStep(prev => Math.max(1, prev - 1))}
          disabled={activeStep === 1}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 transition"
        >
          <ChevronRight className="w-4 h-4" />
          <span>{t.prevStep}</span>
        </button>

        <button
          onClick={onExecuteSingleStep}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-indigo-600 hover:bg-indigo-500 text-white transition shadow-lg shadow-indigo-600/30"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>{lang === 'ar' ? 'تنفيذ خطوة تدريب كاملة وتحديث القيم' : 'Execute Step & Update Values'}</span>
        </button>

        <button
          onClick={() => setActiveStep(prev => Math.min(5, prev + 1))}
          disabled={activeStep === 5}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-700 transition"
        >
          <span>{t.nextStep}</span>
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
