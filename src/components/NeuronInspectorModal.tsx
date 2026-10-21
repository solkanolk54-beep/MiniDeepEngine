import React from 'react';
import { X, Cpu, ArrowRight, Activity, Zap, Layers } from 'lucide-react';
import { NeuralNetwork } from '../engine/model';
import { DenseLayer } from '../engine/layers';
import { Language } from '../translations';

interface NeuronInspectorModalProps {
  model: NeuralNetwork;
  selectedNeuron: { layerIdx: number; neuronIdx: number } | null;
  onClose: () => void;
  lang: Language;
}

export const NeuronInspectorModal: React.FC<NeuronInspectorModalProps> = ({
  model,
  selectedNeuron,
  onClose,
  lang,
}) => {
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!selectedNeuron) return null;

  const { layerIdx, neuronIdx } = selectedNeuron;
  const denseLayers = model.layers.filter(l => l instanceof DenseLayer) as DenseLayer[];

  const isInputLayer = layerIdx === 0;
  const dense = !isInputLayer ? denseLayers[layerIdx - 1] : null;

  // Extract weights feeding into this neuron
  const incomingWeights: { srcIdx: number; w: number; dw?: number; m?: number; v?: number }[] = [];
  let biasVal = 0;
  let dbiasVal = 0;
  let zVal = 0;
  let outVal = 0;

  if (dense) {
    biasVal = dense.biases[0]?.[neuronIdx] ?? 0;
    dbiasVal = dense.dbiases[0]?.[neuronIdx] ?? 0;
    zVal = dense.z?.[0]?.[neuronIdx] ?? 0;
    outVal = dense.output?.[0]?.[neuronIdx] ?? 0;

    for (let i = 0; i < dense.n_inputs; i++) {
      incomingWeights.push({
        srcIdx: i,
        w: dense.weights[i]?.[neuronIdx] ?? 0,
        dw: dense.dweights[i]?.[neuronIdx] ?? 0,
        m: dense.weight_momentums?.[i]?.[neuronIdx] ?? 0,
        v: dense.weight_cache?.[i]?.[neuronIdx] ?? 0,
      });
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-xl max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-indigo-500/10 border border-indigo-500/20 rounded-xl text-indigo-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                {isInputLayer
                  ? lang === 'ar' ? `عصبون المدخل x${neuronIdx + 1}` : `Input Neuron x${neuronIdx + 1}`
                  : lang === 'ar' ? `عصبون الطبقة ${layerIdx} (عصبون #${neuronIdx + 1})` : `Layer ${layerIdx} Neuron #${neuronIdx + 1}`}
              </h3>
              <p className="text-xs text-slate-400">
                {isInputLayer
                  ? lang === 'ar' ? 'عصبون مدخل خام ينقل البيانات إلى الطبقات اللاحقة' : 'Raw input feature passed to subsequent layers'
                  : lang === 'ar' ? 'تشريح المعادلة: z = W • a + b وحسابات التدرج والمشتقة' : 'Detailed equation breakdown: z = W • a + b, gradients & optimizer state'}
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
        <div className="p-5 space-y-5">
          {isInputLayer ? (
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 text-center">
              <p className="text-sm text-slate-300">
                {lang === 'ar'
                  ? 'هذا العصبون ينتمي إلى طبقة المدخلات (Input Features). لا يملك أوزاناً أو انحيازاً خاصاً به، بل يغذي عصبونات الطبقة الأولى بالأوزان.'
                  : 'This neuron represents an input feature coordinate. It has no internal weights or bias, but feeds forward into the first dense layer.'}
              </p>
            </div>
          ) : (
            <>
              {/* Formula & Live Values Box */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3">
                  <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                    {lang === 'ar' ? 'الحساب الخطي (Pre-activation z)' : 'Linear Pre-activation (z)'}
                  </span>
                  <div className="mt-1 flex items-baseline gap-2">
                    <span className="font-mono text-xl font-bold text-cyan-400">
                      {zVal.toFixed(4)}
                    </span>
                    <span className="text-xs text-slate-500 font-mono">z = W·a + b</span>
                  </div>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3">
                  <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">
                    {lang === 'ar' ? 'الانحياز (Bias b) وتدرجه' : 'Bias (b) & Gradient'}
                  </span>
                  <div className="mt-1 flex items-baseline gap-2">
                    <span className="font-mono text-xl font-bold text-emerald-400">
                      {biasVal.toFixed(4)}
                    </span>
                    <span className="text-xs text-rose-400 font-mono">
                      ∂L/∂b: {dbiasVal.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Incoming Weights Table */}
              <div>
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-indigo-400" />
                  <span>{lang === 'ar' ? 'الأوزان الواصلة (Incoming Synaptic Weights)' : 'Incoming Synaptic Weights'}</span>
                </h4>
                <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950/50">
                  <table className="w-full text-xs font-mono text-slate-300">
                    <thead className="bg-slate-800/80 text-slate-400 text-[10px] uppercase border-b border-slate-700/60">
                      <tr>
                        <th className="p-2 text-start">{lang === 'ar' ? 'المصدر' : 'Source'}</th>
                        <th className="p-2 text-start">{lang === 'ar' ? 'الوزن (W)' : 'Weight (W)'}</th>
                        <th className="p-2 text-start">{lang === 'ar' ? 'التدرج (∂L/∂W)' : 'Gradient (∂L/∂W)'}</th>
                        <th className="p-2 text-start">{lang === 'ar' ? 'العزم Adam (m_t)' : 'Momentum (m_t)'}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {incomingWeights.map((syn) => (
                        <tr key={syn.srcIdx} className="hover:bg-slate-800/30">
                          <td className="p-2 text-slate-400">
                            node {syn.srcIdx + 1}
                          </td>
                          <td className={`p-2 font-bold ${syn.w >= 0 ? 'text-cyan-400' : 'text-rose-400'}`}>
                            {syn.w.toFixed(4)}
                          </td>
                          <td className="p-2 text-amber-300">
                            {(syn.dw ?? 0).toFixed(5)}
                          </td>
                          <td className="p-2 text-slate-400">
                            {(syn.m ?? 0).toFixed(5)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Mathematical Insight Note */}
              <div className="bg-indigo-950/40 border border-indigo-900/60 rounded-xl p-3.5 text-xs text-indigo-200 flex gap-2.5">
                <Zap className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <p className="font-semibold text-white">
                    {lang === 'ar' ? 'تحديث خوارزمية Adam لهذا العصبون:' : 'Adam Parameter Update Rule:'}
                  </p>
                  <p className="text-[11px] text-indigo-300 font-mono">
                    W ← W - (η / (sqrt(v̂_t) + ε)) · m̂_t
                  </p>
                  <p className="text-slate-400 text-[11px]">
                    {lang === 'ar'
                      ? 'يقوم محسّن Adam بتعديل وتيرة التعلم لهذا العصبون بناءً على تاريخ تدرجاته السابقة (العزم الأول للتسريع والعزم الثاني لتجنب القفزات المفاجئة).'
                      : 'Adam dynamically adapts the learning step for each parameter using momentum and second-moment variance scaling.'}
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
