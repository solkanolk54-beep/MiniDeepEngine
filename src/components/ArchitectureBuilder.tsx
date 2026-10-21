import React, { useState } from 'react';
import { Layers, Plus, Trash2, Cpu, Settings2, Sliders, CheckCircle2, RotateCcw } from 'lucide-react';
import { NeuralNetwork } from '../engine/model';
import { DenseLayer, DropoutLayer } from '../engine/layers';
import {
  Activation_ReLU,
  Activation_Sigmoid,
  Activation_Tanh,
  Activation_LeakyReLU,
  Activation_Linear,
  LayerInterface,
} from '../engine/activations';
import { Loss_MSE, Loss_BinaryCrossEntropy, LossInterface } from '../engine/loss';
import { Optimizer_Adam, Optimizer_SGD, OptimizerInterface } from '../engine/optimizers';
import { Language, translations } from '../translations';

export interface LayerConfig {
  id: string;
  type: 'dense' | 'activation' | 'dropout';
  neurons?: number;
  activationType?: 'relu' | 'sigmoid' | 'tanh' | 'leaky_relu' | 'linear';
  leakyAlpha?: number;
  dropoutRate?: number;
}

interface ArchitectureBuilderProps {
  inputDim: number;
  outputDim: number;
  onApplyArchitecture: (layers: LayerInterface[], loss: LossInterface, optimizer: OptimizerInterface) => void;
  lang: Language;
}

export const ArchitectureBuilder: React.FC<ArchitectureBuilderProps> = ({
  inputDim,
  outputDim,
  onApplyArchitecture,
  lang,
}) => {
  const t = translations[lang];

  // Configurable layers list
  const [layersConfig, setLayersConfig] = useState<LayerConfig[]>([
    { id: 'l1', type: 'dense', neurons: 8 },
    { id: 'l2', type: 'activation', activationType: 'relu' },
    { id: 'l3', type: 'dense', neurons: outputDim },
    { id: 'l4', type: 'activation', activationType: 'sigmoid' },
  ]);

  // Optimizer & Loss settings
  const [optimizerType, setOptimizerType] = useState<'adam' | 'sgd'>('adam');
  const [learningRate, setLearningRate] = useState<number>(0.02);
  const [beta1, setBeta1] = useState<number>(0.9);
  const [beta2, setBeta2] = useState<number>(0.999);
  const [momentum, setMomentum] = useState<number>(0.0);
  const [lossType, setLossType] = useState<'mse' | 'bce'>('mse');

  const addDenseLayer = () => {
    // Insert before the last output layer
    const newId = `l_${Date.now()}`;
    setLayersConfig(prev => {
      const copy = [...prev];
      copy.splice(Math.max(0, copy.length - 2), 0,
        { id: `${newId}_d`, type: 'dense', neurons: 8 },
        { id: `${newId}_a`, type: 'activation', activationType: 'relu' }
      );
      return copy;
    });
  };

  const removeLayer = (id: string) => {
    if (layersConfig.length <= 2) return;
    setLayersConfig(prev => prev.filter(l => l.id !== id));
  };

  const updateLayerNeurons = (id: string, count: number) => {
    setLayersConfig(prev =>
      prev.map(l => (l.id === id ? { ...l, neurons: Math.max(1, count) } : l))
    );
  };

  const updateActivationType = (id: string, act: LayerConfig['activationType']) => {
    setLayersConfig(prev =>
      prev.map(l => (l.id === id ? { ...l, activationType: act } : l))
    );
  };

  const handleApply = () => {
    const builtLayers: LayerInterface[] = [];
    let currentInDim = inputDim;

    for (let i = 0; i < layersConfig.length; i++) {
      const cfg = layersConfig[i];
      if (cfg.type === 'dense') {
        const neurons = cfg.neurons || 4;
        builtLayers.push(new DenseLayer(currentInDim, neurons));
        currentInDim = neurons;
      } else if (cfg.type === 'activation') {
        switch (cfg.activationType) {
          case 'relu':
            builtLayers.push(new Activation_ReLU());
            break;
          case 'sigmoid':
            builtLayers.push(new Activation_Sigmoid());
            break;
          case 'tanh':
            builtLayers.push(new Activation_Tanh());
            break;
          case 'leaky_relu':
            builtLayers.push(new Activation_LeakyReLU(cfg.leakyAlpha || 0.01));
            break;
          case 'linear':
          default:
            builtLayers.push(new Activation_Linear());
            break;
        }
      } else if (cfg.type === 'dropout') {
        builtLayers.push(new DropoutLayer(cfg.dropoutRate || 0.2));
      }
    }

    const lossObj: LossInterface =
      lossType === 'bce' ? new Loss_BinaryCrossEntropy() : new Loss_MSE();

    const optObj: OptimizerInterface =
      optimizerType === 'adam'
        ? new Optimizer_Adam(learningRate, beta1, beta2)
        : new Optimizer_SGD(learningRate, momentum);

    onApplyArchitecture(builtLayers, lossObj, optObj);
  };

  const applyPreset = (preset: 'prompt' | 'deep' | 'xor') => {
    if (preset === 'prompt') {
      setLayersConfig([
        { id: 'p1', type: 'dense', neurons: 8 },
        { id: 'p2', type: 'activation', activationType: 'relu' },
        { id: 'p3', type: 'dense', neurons: outputDim },
      ]);
      setOptimizerType('adam');
      setLearningRate(0.01);
      setLossType('mse');
    } else if (preset === 'deep') {
      setLayersConfig([
        { id: 'd1', type: 'dense', neurons: 16 },
        { id: 'd2', type: 'activation', activationType: 'relu' },
        { id: 'd3', type: 'dense', neurons: 16 },
        { id: 'd4', type: 'activation', activationType: 'relu' },
        { id: 'd5', type: 'dense', neurons: 8 },
        { id: 'd6', type: 'activation', activationType: 'tanh' },
        { id: 'd7', type: 'dense', neurons: outputDim },
        { id: 'd8', type: 'activation', activationType: 'sigmoid' },
      ]);
      setOptimizerType('adam');
      setLearningRate(0.02);
      setLossType('mse');
    } else if (preset === 'xor') {
      setLayersConfig([
        { id: 'x1', type: 'dense', neurons: 8 },
        { id: 'x2', type: 'activation', activationType: 'relu' },
        { id: 'x3', type: 'dense', neurons: 1 },
        { id: 'x4', type: 'activation', activationType: 'sigmoid' },
      ]);
      setOptimizerType('adam');
      setLearningRate(0.05);
      setLossType('mse');
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Title */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            <span>{t.architectureTab}</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            {lang === 'ar'
              ? 'تخصيص عدد الطبقات والعصبونات ودوال التفعيل مع ربط فوري بكود بايثون'
              : 'Configure layers, neurons count, activations and hyper-parameters with instant synchronization'}
          </p>
        </div>

        {/* Preset Architectures */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-400 hidden sm:inline">{t.presetsTitle}:</span>
          <button
            onClick={() => applyPreset('prompt')}
            className="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            {lang === 'ar' ? 'نموذج المقال (2-8-1)' : 'Prompt (2-8-1)'}
          </button>
          <button
            onClick={() => applyPreset('xor')}
            className="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700"
          >
            XOR Solver
          </button>
          <button
            onClick={() => applyPreset('deep')}
            className="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700"
          >
            Deep (16-16-8-1)
          </button>
        </div>
      </div>

      {/* Layers List */}
      <div className="space-y-3">
        <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span>{t.layersTitle}</span>
        </h3>

        <div className="space-y-2">
          {layersConfig.map((layer, idx) => (
            <div
              key={layer.id}
              className="flex items-center justify-between gap-3 bg-slate-950 border border-slate-800 p-3 rounded-xl hover:border-slate-700 transition"
            >
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-400 font-mono text-xs flex items-center justify-center font-bold">
                  {idx + 1}
                </span>

                {layer.type === 'dense' && (
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-xs font-bold text-cyan-400 uppercase font-mono">
                      Dense Layer
                    </span>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <span>{t.neurons}:</span>
                      <input
                        type="number"
                        min={1}
                        max={64}
                        value={layer.neurons || 4}
                        onChange={(e) => updateLayerNeurons(layer.id, parseInt(e.target.value) || 1)}
                        className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono text-center focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  </div>
                )}

                {layer.type === 'activation' && (
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-xs font-bold text-emerald-400 uppercase font-mono">
                      Activation
                    </span>
                    <select
                      value={layer.activationType || 'relu'}
                      onChange={(e) => updateActivationType(layer.id, e.target.value as any)}
                      className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white focus:outline-none cursor-pointer"
                    >
                      <option value="relu">ReLU (max(0, x))</option>
                      <option value="sigmoid">Sigmoid (1 / (1 + e⁻ˣ))</option>
                      <option value="tanh">Tanh (tanh(x))</option>
                      <option value="leaky_relu">Leaky ReLU (α=0.01)</option>
                      <option value="linear">Linear (Identity)</option>
                    </select>
                  </div>
                )}
              </div>

              {layersConfig.length > 2 && (
                <button
                  onClick={() => removeLayer(layer.id)}
                  title={t.removeLayer}
                  className="p-1.5 text-slate-500 hover:text-rose-400 rounded-lg hover:bg-slate-900 transition"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}
        </div>

        <button
          onClick={addDenseLayer}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium text-indigo-400 hover:text-indigo-300 bg-indigo-950/40 hover:bg-indigo-950/70 border border-indigo-800/50 transition w-full justify-center"
        >
          <Plus className="w-4 h-4" />
          <span>{t.addDense}</span>
        </button>
      </div>

      {/* Optimizer & Loss Grid */}
      <div className="grid md:grid-cols-2 gap-4 pt-2">
        {/* Optimizer Settings */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h4 className="text-xs font-bold text-white">{t.optimizerSettings}</h4>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setOptimizerType('adam')}
              className={`flex-1 py-1.5 rounded-lg text-xs font-bold transition ${
                optimizerType === 'adam'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              Adam (Adaptive Moments)
            </button>
            <button
              onClick={() => setOptimizerType('sgd')}
              className={`flex-1 py-1.5 rounded-lg text-xs font-bold transition ${
                optimizerType === 'sgd'
                  ? 'bg-cyan-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              SGD (Gradient Descent)
            </button>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between items-center text-slate-400">
              <span>{t.learningRate} (η):</span>
              <input
                type="number"
                step="0.005"
                min="0.001"
                max="0.5"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono text-center"
              />
            </div>

            {optimizerType === 'adam' && (
              <>
                <div className="flex justify-between items-center text-slate-400">
                  <span>β₁ (First Moment):</span>
                  <input
                    type="number"
                    step="0.01"
                    min="0.5"
                    max="0.99"
                    value={beta1}
                    onChange={(e) => setBeta1(parseFloat(e.target.value) || 0.9)}
                    className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono text-center"
                  />
                </div>
                <div className="flex justify-between items-center text-slate-400">
                  <span>β₂ (Second Moment):</span>
                  <input
                    type="number"
                    step="0.001"
                    min="0.9"
                    max="0.9999"
                    value={beta2}
                    onChange={(e) => setBeta2(parseFloat(e.target.value) || 0.999)}
                    className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono text-center"
                  />
                </div>
              </>
            )}

            {optimizerType === 'sgd' && (
              <div className="flex justify-between items-center text-slate-400">
                <span>Momentum:</span>
                <input
                  type="number"
                  step="0.1"
                  min="0.0"
                  max="0.9"
                  value={momentum}
                  onChange={(e) => setMomentum(parseFloat(e.target.value) || 0)}
                  className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono text-center"
                />
              </div>
            )}
          </div>
        </div>

        {/* Loss Function Settings */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-emerald-400" />
            <h4 className="text-xs font-bold text-white">{t.lossSettings}</h4>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setLossType('mse')}
              className={`flex-1 py-1.5 rounded-lg text-xs font-bold transition ${
                lossType === 'mse'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              Mean Squared Error (MSE)
            </button>
            <button
              onClick={() => setLossType('bce')}
              className={`flex-1 py-1.5 rounded-lg text-xs font-bold transition ${
                lossType === 'bce'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              Binary Cross-Entropy
            </button>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed pt-2">
            {lossType === 'mse'
              ? lang === 'ar'
                ? 'تُحسب كمتوسط لمربعات الفروق. ملائمة لجميع مسائل الانحدار والتصنيف البسيط مع دوال التفعيل.'
                : 'Computes mean of squared differences. Well-suited for regression and general function learning.'
              : lang === 'ar'
                ? 'دالة الإنتروبيا المتقاطعة للمشاكل الثنائية، تعاقب التوقعات الخاطئة بشدة وتسرع تقارب الاحتمالات.'
                : 'Penalizes confident wrong classifications heavily, ideal for probability distribution learning.'}
          </p>
        </div>
      </div>

      {/* Apply Button */}
      <div className="pt-2">
        <button
          onClick={handleApply}
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-bold text-sm bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-lg shadow-indigo-600/30 transition active:scale-[0.99]"
        >
          <CheckCircle2 className="w-4 h-4" />
          <span>{t.applyArchitecture}</span>
        </button>
      </div>
    </div>
  );
};
