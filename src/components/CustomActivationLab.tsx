/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, Check, AlertCircle, ArrowUpRight, HelpCircle } from 'lucide-react';
import { Activation_Custom, CustomActivationType } from '../engine/customActivation';
import { Language, translations } from '../translations';
import { NeuralNetwork } from '../engine/model';

type ActivationPresetKey = 'swish' | 'mish' | 'gelu' | 'siren' | 'softplus' | 'custom';

interface CustomActivationLabProps {
  model: NeuralNetwork;
  lang: Language;
  onActivationInjected: (activationName: string) => void;
}

export const CustomActivationLab: React.FC<CustomActivationLabProps> = ({
  model,
  lang,
  onActivationInjected,
}) => {
  const t = translations[lang];
  const [selectedPreset, setSelectedPreset] = useState<ActivationPresetKey>('swish');
  const [customFormula, setCustomFormula] = useState<string>('x * Math.tanh(Math.log(1 + Math.exp(x)))');
  const [testInputVal, setTestInputVal] = useState<number>(1.5);
  const [injectedSuccess, setInjectedSuccess] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const actType: CustomActivationType = selectedPreset === 'custom' ? 'custom_expr' : selectedPreset;

  // Active custom activation instance
  const activeCustomAct = new Activation_Custom(
    actType,
    customFormula
  );

  // Preset mathematical definitions & descriptions
  const presetDetails: Record<
    ActivationPresetKey,
    { name: string; formula: string; derivative: string; descAr: string; descEn: string; inventedBy: string }
  > = {
    swish: {
      name: 'Swish (SiLU)',
      formula: 'f(x) = x • σ(x)',
      derivative: "f'(x) = σ(x) + x • σ(x) • (1 - σ(x))",
      descAr: 'دالة تفعيل ناعمة غير رتيبة اكتشفتها Google Brain عبر البحث التلقائي، تتفوق غالباً على ReLU في الشبكات العميقة.',
      descEn: 'Smooth, non-monotonic self-gated activation found via neural architecture search by Google Brain.',
      inventedBy: 'Google Brain (2017)',
    },
    mish: {
      name: 'Mish',
      formula: 'f(x) = x • tanh(ln(1 + eˣ))',
      derivative: "f'(x) = Δ(x) • sech²(ln(1 + eˣ)) + tanh(ln(1 + eˣ))",
      descAr: 'دالة تفعيل متصلة وناعمة تمنع تشبع التدرجات وتسمح بتدفق المعلومات السلبية الصغيرة لتجنب موت العصبونات.',
      descEn: 'Self-regularized smooth non-monotonic activation function widely used in computer vision (YOLOv4).',
      inventedBy: 'Diganta Misra (2019)',
    },
    gelu: {
      name: 'GELU (Gaussian Error Linear Unit)',
      formula: 'f(x) = 0.5x • (1 + tanh(√(2/π)(x + 0.044715x³)))',
      derivative: "f'(x) = 0.5(1 + erf(x/√2)) + x • N(x;0,1)",
      descAr: 'المعيار الذهبي في معمارية Transformers ونماذج اللغة الكبرى (BERT, GPT, Gemini). تزن المدخلات بقيمتها الاحتمالية.',
      descEn: 'Gold standard for Transformers & modern LLMs. Weights inputs by their probabilistic CDF.',
      inventedBy: 'Hendrycks & Gimpel (2016)',
    },
    siren: {
      name: 'SIREN (Sinusoidal Representation)',
      formula: 'f(x) = sin(30 • x)',
      derivative: "f'(x) = 30 • cos(30 • x)",
      descAr: 'تستخدم دوال الجيب الدائرية لتمثيل الإشارات المعقدة، والتضاريس ثلاثية الأبعاد (NeRFs)، ومشتقاتها ذات رتبة عليا لا تتلاشى.',
      descEn: 'Periodic sinusoidal activation ideal for implicit neural representations (NeRFs) and physics PINNs.',
      inventedBy: 'Sitzmann et al. (2020)',
    },
    softplus: {
      name: 'Softplus',
      formula: 'f(x) = ln(1 + eˣ)',
      derivative: "f'(x) = σ(x) = 1 / (1 + e⁻ˣ)",
      descAr: 'التقريب الرياضي الناعم لدالة ReLU، ومشتقتها التحليلية المباشرة هي دالة السيجمويد Sigmoid!',
      descEn: 'Smooth continuous approximation of ReLU whose exact analytical derivative is Sigmoid.',
      inventedBy: 'Dugas et al. (2001)',
    },
    custom: {
      name: lang === 'ar' ? 'معادلة رياضية خاصة' : 'Custom User Formula',
      formula: 'f(x) = (Custom JS Math)',
      derivative: "f'(x) = (Finite Numerical Difference)",
      descAr: 'اكتب معادلتك الرياضية الخاصة واختبر انسيابية التدرج الرياضي مباشرة وتأثيرها على تدريب الشبكة.',
      descEn: 'Write any custom mathematical expression and immediately test its gradient flow and loss curve.',
      inventedBy: 'You (R&D)',
    },
  };

  // Render Canvas Plot: f(x) and f'(x)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Coordinate mapping: x in [-4, 4], y in [-3, 3]
    const xMin = -4;
    const xMax = 4;
    const yMin = -2.5;
    const yMax = 3.5;

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height;

    // Draw Grid & Axes
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let x = -4; x <= 4; x += 1) {
      const cx = toCanvasX(x);
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, height);
      ctx.stroke();
    }
    for (let y = -2; y <= 3; y += 1) {
      const cy = toCanvasY(y);
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(width, cy);
      ctx.stroke();
    }

    // Origin Axes
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.5;
    // X Axis
    const originY = toCanvasY(0);
    ctx.beginPath();
    ctx.moveTo(0, originY);
    ctx.lineTo(width, originY);
    ctx.stroke();
    // Y Axis
    const originX = toCanvasX(0);
    ctx.beginPath();
    ctx.moveTo(originX, 0);
    ctx.lineTo(originX, height);
    ctx.stroke();

    // Plot Function f(x) (Cyan Curve)
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    const step = 0.05;
    for (let x = xMin; x <= xMax; x += step) {
      const y = activeCustomAct.evalScalar(x);
      if (isNaN(y) || !isFinite(y)) continue;
      const cx = toCanvasX(x);
      const cy = Math.max(-50, Math.min(height + 50, toCanvasY(y)));
      if (!started) {
        ctx.moveTo(cx, cy);
        started = true;
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();

    // Plot Derivative f'(x) (Amber Curve)
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    started = false;
    for (let x = xMin; x <= xMax; x += step) {
      const dy = activeCustomAct.evalDerivativeScalar(x);
      if (isNaN(dy) || !isFinite(dy)) continue;
      const cx = toCanvasX(x);
      const cy = Math.max(-50, Math.min(height + 50, toCanvasY(dy)));
      if (!started) {
        ctx.moveTo(cx, cy);
        started = true;
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Highlight Current Test Point
    const testY = activeCustomAct.evalScalar(testInputVal);
    const ptX = toCanvasX(testInputVal);
    const ptY = toCanvasY(testY);

    if (isFinite(testY)) {
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(ptX, ptY, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [selectedPreset, customFormula, testInputVal]);

  const handleInject = () => {
    // Inject this custom activation into the model's activation layers
    let injectedCount = 0;
    model.layers.forEach((layer: any) => {
      // If it's an activation layer or custom layer
      if (layer instanceof Activation_Custom || layer.constructor.name.startsWith('Activation')) {
        const replacement = new Activation_Custom(actType, customFormula);
        Object.assign(layer, replacement);
        injectedCount++;
      }
    });

    if (injectedCount > 0) {
      setInjectedSuccess(true);
      onActivationInjected(presetDetails[selectedPreset].name);
      setTimeout(() => setInjectedSuccess(false), 3000);
    }
  };

  const testOutput = activeCustomAct.evalScalar(testInputVal);
  const testGrad = activeCustomAct.evalDerivativeScalar(testInputVal);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col h-full">
      {/* Header Bar */}
      <div className="bg-slate-950 px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <span>{t.customActivationTitle}</span>
              <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/40 px-2 py-0.5 rounded-full">
                R&D Sandbox
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              {lang === 'ar'
                ? 'ابتكار دوال تفعيل جديدة واختبار مشتقاتها التحليلية وتأثيرها على تدفق التدرجات'
                : 'Invent novel activations, visualize analytic derivatives, and inject directly into training'}
            </p>
          </div>
        </div>

        {/* Inject Trigger */}
        <button
          onClick={handleInject}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white shadow-lg shadow-cyan-950/50 transition-all hover:scale-[1.02] active:scale-[0.98]"
        >
          {injectedSuccess ? <Check className="w-4 h-4 text-emerald-300" /> : <ArrowUpRight className="w-4 h-4" />}
          <span>{injectedSuccess ? t.activationInjected : t.injectActivation}</span>
        </button>
      </div>

      <div className="p-4 grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1 overflow-auto">
        {/* Left Column: Preset Selector & Formula Details */}
        <div className="lg:col-span-5 space-y-4">
          {/* Preset Buttons */}
          <div>
            <label className="text-xs font-semibold text-slate-300 mb-2 block">
              {lang === 'ar' ? 'اختر دالة تفعيل رائدة أو اكتب معادلتك:' : 'Select Breakthrough Preset or Custom:'}
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(['swish', 'mish', 'gelu', 'siren', 'softplus', 'custom'] as ActivationPresetKey[]).map((preset) => (
                <button
                  key={preset}
                  onClick={() => setSelectedPreset(preset)}
                  className={`text-left p-2 rounded-lg text-xs font-mono transition-all border ${
                    selectedPreset === preset
                      ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200 font-bold shadow-md shadow-cyan-950/30'
                      : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span>{presetDetails[preset].name}</span>
                    {selectedPreset === preset && <Check className="w-3.5 h-3.5 text-cyan-400" />}
                  </div>
                  <div className="text-[10px] text-slate-500 font-sans mt-0.5 truncate">
                    {presetDetails[preset].inventedBy}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Formula Details Box */}
          <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2.5 font-mono text-xs" dir="ltr">
            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Forward Mapping f(x)</div>
              <div className="text-cyan-400 font-bold text-sm mt-0.5">{presetDetails[selectedPreset].formula}</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Analytic Derivative f&apos;(x)</div>
              <div className="text-amber-400 font-semibold text-xs mt-0.5">{presetDetails[selectedPreset].derivative}</div>
            </div>
            <p className="text-slate-400 font-sans text-xs pt-1 border-t border-slate-800/80 leading-relaxed" dir={lang === 'ar' ? 'rtl' : 'ltr'}>
              {lang === 'ar' ? presetDetails[selectedPreset].descAr : presetDetails[selectedPreset].descEn}
            </p>
          </div>

          {/* Custom Formula Editor if selected */}
          {selectedPreset === 'custom' && (
            <div className="bg-slate-950 p-3 rounded-lg border border-cyan-800/60 space-y-2">
              <label className="text-xs font-semibold text-cyan-300 flex items-center justify-between">
                <span>{lang === 'ar' ? 'صيغة التفعيل في جافاسكريبت:' : 'JavaScript Activation Expression:'}</span>
                <span className="text-[10px] font-mono text-slate-400">variable: x</span>
              </label>
              <input
                type="text"
                value={customFormula}
                onChange={(e) => setCustomFormula(e.target.value)}
                placeholder="e.g. x * Math.tanh(x)"
                className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-xs font-mono text-cyan-200 focus:outline-none focus:border-cyan-400"
                dir="ltr"
              />
              <div className="text-[11px] text-slate-400 flex items-center gap-1">
                <HelpCircle className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                <span>
                  {lang === 'ar'
                    ? 'يتم حساب المشتقة رقمياً تلقائياً بواسطة Finite Differences'
                    : 'Derivative is computed automatically via Central Finite Differences'}
                </span>
              </div>
            </div>
          )}

          {/* Interactive Scalar Test Evaluation */}
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">
                {lang === 'ar' ? 'تجربة قيمة مدخلة (Scalar Probe):' : 'Scalar Probe Value (x):'}
              </span>
              <span className="font-mono text-cyan-400 font-bold">{testInputVal.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-4"
              max="4"
              step="0.1"
              value={testInputVal}
              onChange={(e) => setTestInputVal(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-xs" dir="ltr">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                <div className="text-[10px] text-cyan-400/80 uppercase">f({testInputVal.toFixed(2)})</div>
                <div className="text-sm font-bold text-cyan-300">{testOutput.toFixed(4)}</div>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                <div className="text-[10px] text-amber-400/80 uppercase">f&apos;({testInputVal.toFixed(2)})</div>
                <div className="text-sm font-bold text-amber-300">{testGrad.toFixed(4)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Dynamic Plot Canvas */}
        <div className="lg:col-span-7 flex flex-col space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-slate-300 flex items-center gap-2">
              <span>{t.curveComparison}</span>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono" dir="ltr">
              <span className="flex items-center gap-1.5 text-cyan-400">
                <span className="w-3 h-0.5 bg-cyan-400 rounded-full inline-block"></span> f(x)
              </span>
              <span className="flex items-center gap-1.5 text-amber-400">
                <span className="w-3 h-0.5 bg-amber-400 border-b border-dashed inline-block"></span> f&apos;(x) [Gradient]
              </span>
            </div>
          </div>

          <div className="bg-black/95 border border-slate-800 rounded-lg p-2 relative flex-1 flex items-center justify-center">
            <canvas
              ref={canvasRef}
              width={560}
              height={320}
              className="w-full max-w-full h-auto rounded"
            />
          </div>

          {/* Mathematical Insights Bar */}
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs text-slate-400 flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" />
            <div className="leading-relaxed">
              <strong className="text-slate-200">
                {lang === 'ar' ? 'فائدة بحثية (Gradient Flow): ' : 'Research Insight: '}
              </strong>
              {testGrad === 0 ? (
                <span className="text-rose-400">
                  {lang === 'ar'
                    ? 'المشتقة تساوي صفراً في هذه النقطة! قد يؤدي ذلك إلى مشكلة العصبون الميت (Dying Neuron).'
                    : 'Gradient is exactly 0 here (Dead Neuron effect!). Upstream loss cannot propagate.'}
                </span>
              ) : Math.abs(testGrad) > 5 ? (
                <span className="text-amber-400">
                  {lang === 'ar'
                    ? 'المشتقة مرتفعة جداً! قد يؤدي ذلك إلى انفجار التدرجات (Exploding Gradients).'
                    : 'Gradient is very steep! Might lead to exploding gradients without clipping.'}
                </span>
              ) : (
                <span className="text-emerald-400">
                  {lang === 'ar'
                    ? 'التدرج متوازن وسلس! يسمح للانتشار الخلفي بنقل إشارات الخطأ بانسيابية إلى الطبقات الأولى.'
                    : 'Smooth healthy gradient flow. Backpropagation will propagate loss stably across hidden layers.'}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
