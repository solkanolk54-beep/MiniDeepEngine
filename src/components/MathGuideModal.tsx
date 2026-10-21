import React from 'react';
import { X, BookOpen, Layers, Zap, TrendingDown, GitBranch, Cpu, ShieldCheck } from 'lucide-react';
import { Language } from '../translations';

interface MathGuideModalProps {
  lang: Language;
  onClose: () => void;
}

export const MathGuideModal: React.FC<MathGuideModalProps> = ({ lang, onClose }) => {
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">
                {lang === 'ar' ? 'الدليل الرياضي والمفاهيمي لمنظومة MiniDeepEngine' : 'MiniDeepEngine Mathematical & Conceptual Reference'}
              </h3>
              <p className="text-xs text-slate-400">
                {lang === 'ar'
                  ? 'شرح معمق لكل مفهوم رياضي تم استخدامه في بناء النواة من الصفر'
                  : 'In-depth breakdown of every mathematical building block implemented from scratch'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm text-slate-300 leading-relaxed">
          {/* Section 1 */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-base font-bold text-indigo-400 flex items-center gap-2">
              <Layers className="w-5 h-5" />
              <span>1. {lang === 'ar' ? 'طبقة العصبون والشبكة الأمامية (Neurons & Feedforward)' : 'Neurons & Feedforward Pass'}</span>
            </h4>
            <p>
              {lang === 'ar'
                ? 'العصبون الاصطناعي يستقبل المدخلات ويجري عملية ضرب قياسي (Dot Product) مع أوزان الاتصال ثم يضيف حد الانحياز (Bias):'
                : 'Each artificial neuron calculates an inner dot product between its input vector and synaptic weights, followed by an additive bias:'}
            </p>
            <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl font-mono text-cyan-300 text-center font-bold">
              z = W • a + b
            </div>
            <p className="text-xs text-slate-400">
              {lang === 'ar'
                ? 'حيث W مصفوفة الأوزان بأبعاد (n_inputs, n_neurons)، و a متجه المدخلات، و b متجه الانحياز بأبعاد (1, n_neurons).'
                : 'Where W is the weight matrix (n_inputs x n_neurons), a is activation input, and b is the bias vector (1 x n_neurons).'}
            </p>
          </div>

          {/* Section 2 */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-base font-bold text-violet-400 flex items-center gap-2">
              <Zap className="w-5 h-5" />
              <span>2. {lang === 'ar' ? 'دوال التفعيل ومشتقاتها (Activation Functions & Derivatives)' : 'Activation Functions & Analytical Derivatives'}</span>
            </h4>
            <p>
              {lang === 'ar'
                ? 'دوال التفعيل تحوّل المخرج الخطي إلى لاخطي a = σ(z)، ومشتقتها ضرورية لحساب التدرج أثناء الانتشار الخلفي:'
                : 'Non-linear activations allow networks to model non-linear boundaries. Their exact analytical derivatives drive backpropagation:'}
            </p>
            <div className="grid sm:grid-cols-2 gap-3 text-xs font-mono">
              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">ReLU (Rectified Linear Unit)</span>
                <p className="text-slate-200">Forward: f(z) = max(0, z)</p>
                <p className="text-emerald-400 mt-1">Backward: f'(z) = 1 if z &gt; 0 else 0</p>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">Sigmoid Function</span>
                <p className="text-slate-200">Forward: σ(z) = 1 / (1 + e⁻ᶻ)</p>
                <p className="text-emerald-400 mt-1">Backward: σ'(z) = σ(z) · (1 - σ(z))</p>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">Tanh (Hyperbolic Tangent)</span>
                <p className="text-slate-200">Forward: tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)</p>
                <p className="text-emerald-400 mt-1">Backward: tanh'(z) = 1 - tanh²(z)</p>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <span className="text-indigo-400 font-bold block mb-1">Leaky ReLU</span>
                <p className="text-slate-200">Forward: f(z) = z if z &gt; 0 else α·z</p>
                <p className="text-emerald-400 mt-1">Backward: f'(z) = 1 if z &gt; 0 else α</p>
              </div>
            </div>
          </div>

          {/* Section 3 */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-base font-bold text-cyan-400 flex items-center gap-2">
              <TrendingDown className="w-5 h-5" />
              <span>3. {lang === 'ar' ? 'دوال الخسارة (Loss Functions)' : 'Loss Functions (MSE & Cross-Entropy)'}</span>
            </h4>
            <div className="grid sm:grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-2">
                <span className="text-cyan-400 font-bold font-mono text-sm">MSE (Mean Squared Error)</span>
                <p className="text-slate-300 font-mono">Loss = (1/N) · Σ (ŷᵢ - yᵢ)²</p>
                <p className="text-amber-400 font-mono">Gradient: ∂L/∂ŷ = (2/N) · (ŷᵢ - yᵢ)</p>
                <p className="text-slate-400 text-[11px]">
                  {lang === 'ar' ? 'تُستخدم للمسائل العددية ومشاكل الانحدار (Regression).' : 'Standard loss function for numeric regression problems.'}
                </p>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl space-y-2">
                <span className="text-emerald-400 font-bold font-mono text-sm">Binary Cross-Entropy</span>
                <p className="text-slate-300 font-mono">Loss = -(1/N) · Σ [y ln(p) + (1-y) ln(1-p)]</p>
                <p className="text-amber-400 font-mono">Gradient: ∂L/∂p = -[y/p - (1-y)/(1-p)] / N</p>
                <p className="text-slate-400 text-[11px]">
                  {lang === 'ar' ? 'المعيار الرياضي الأمثل لمشاكل التصنيف الثنائي (Classification).' : 'Optimal choice for binary classification and probabilities.'}
                </p>
              </div>
            </div>
          </div>

          {/* Section 4 */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-base font-bold text-amber-400 flex items-center gap-2">
              <GitBranch className="w-5 h-5" />
              <span>4. {lang === 'ar' ? 'الانتشار الخلفي وقاعدة السلسلة (Backpropagation & Chain Rule)' : 'Backpropagation & Chain Rule'}</span>
            </h4>
            <p>
              {lang === 'ar'
                ? 'نطبق قاعدة السلسلة الرياضية لنشتق دالة الخسارة بالنسبة لكل وزن وانحياز ومدخل:'
                : 'Applying the multivariate Chain Rule to calculate derivatives of Loss with respect to every weight and bias:'}
            </p>
            <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl font-mono text-xs space-y-1 text-slate-200">
              <p>1. ∂L/∂W = inputsᵀ • dvalues</p>
              <p>2. ∂L/∂b = Σ dvalues (axis = 0)</p>
              <p>3. ∂L/∂inputs = dvalues • weightsᵀ</p>
            </div>
          </div>

          {/* Section 5 */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-base font-bold text-emerald-400 flex items-center gap-2">
              <Cpu className="w-5 h-5" />
              <span>5. {lang === 'ar' ? 'خوارزمية آدم (Adam Optimizer)' : 'Adam (Adaptive Moment Estimation) Optimizer'}</span>
            </h4>
            <p>
              {lang === 'ar'
                ? 'تعتبر من أكثر الخوارزميات كفاءة في التعلم العميق لأنها تجمع بين Momentum و RMSprop مع تصحيح الانحياز:'
                : 'Adam combines Momentum (exponential moving average of gradients) and RMSprop (moving average of squared gradients) with initial bias correction:'}
            </p>
            <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl font-mono text-xs space-y-1.5 text-cyan-300">
              <p>mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ  (العزم الأول - First Moment)</p>
              <p>vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²  (العزم الثاني - Second Moment)</p>
              <p className="text-amber-300">m̂ₜ = mₜ / (1 - β₁ᵗ)  (تصحيح انحياز العزم الأول)</p>
              <p className="text-amber-300">v̂ₜ = vₜ / (1 - β₂ᵗ)  (تصحيح انحياز العزم الثاني)</p>
              <p className="text-emerald-400 font-bold">θ ← θ - [ η / (√v̂ₜ + ε) ] · m̂ₜ  (التحديث النهائي للمتغيرات)</p>
            </div>
          </div>

          {/* Section 6 */}
          <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-base font-bold text-pink-400 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5" />
              <span>6. {lang === 'ar' ? 'التنظيم والتقييم (Regularization & Metrics)' : 'Regularization & Evaluation Metrics'}</span>
            </h4>
            <p>
              {lang === 'ar'
                ? 'تقنيات لمنع فرط التخصيص (Overfitting) وتقييم الأداء:'
                : 'Techniques to prevent overfitting and measure generalization:'}
            </p>
            <ul className="list-disc list-inside space-y-1 text-xs text-slate-400">
              <li><strong className="text-slate-200">L2 Regularization (Weight Decay):</strong> إضافة عقوبة لمربعات الأوزان في دالة الخسارة وتدرجها 2·λ·W لمنع الأوزان الضخمة.</li>
              <li><strong className="text-slate-200">Dropout:</strong> تعطيل نسبة عشوائية من العصبونات أثناء التدريب لفرض استقلالية السمات.</li>
              <li><strong className="text-slate-200">Accuracy & F1-Score:</strong> قياس الدقة ومعدل التوازن بين الحساسية والدقة للتصنيف.</li>
              <li><strong className="text-slate-200">RMSE:</strong> جذر متوسط مربع الخطأ لتقييم مقدار البعد عن القيمة الحقيقية في الانحدار.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
