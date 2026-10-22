/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Copy, Check, Download, Trash2, ArrowRight, Activity } from 'lucide-react';
import { MathInspectorLogger as EngineLogger, InspectorEpochLog, InspectorLayerLog } from '../engine/inspectorLogger';
import { Language, translations } from '../translations';
import { NeuralNetwork } from '../engine/model';

interface MathInspectorLoggerProps {
  logger: EngineLogger;
  model: NeuralNetwork;
  lang: Language;
  onClearLogs: () => void;
}

export const MathInspectorLogger: React.FC<MathInspectorLoggerProps> = ({
  logger,
  model,
  lang,
  onClearLogs,
}) => {
  const t = translations[lang];
  const [copied, setCopied] = useState(false);
  const [selectedEpochIdx, setSelectedEpochIdx] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'terminal' | 'visual_cards'>('terminal');
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const snapshots: InspectorEpochLog[] = logger.getSnapshots();
  const rawTerminalText: string = logger.getFormattedLogs();

  // Keep latest epoch active by default if not manually chosen
  const activeSnapshot: InspectorEpochLog | undefined =
    selectedEpochIdx !== null && snapshots[selectedEpochIdx]
      ? snapshots[selectedEpochIdx]
      : snapshots[snapshots.length - 1];

  useEffect(() => {
    if (viewMode === 'terminal' && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [snapshots.length, viewMode]);

  const handleCopy = () => {
    navigator.clipboard.writeText(rawTerminalText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadPythonInspector = () => {
    const code = model.exportInspectorPythonCode();
    const blob = new Blob([code], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'minideepengine_inspector.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  const getMatrixNorm = (m?: number[][]): number => {
    if (!m || m.length === 0) return 0;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < m.length; i++) {
      for (let j = 0; j < m[i].length; j++) {
        sum += m[i][j] * m[i][j];
        count++;
      }
    }
    return count > 0 ? Math.sqrt(sum) : 0;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col h-full">
      {/* Header Bar */}
      <div className="bg-slate-950 px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Terminal className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <span>{t.mathLoggerTitle}</span>
              <span className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                {snapshots.length} {snapshots.length === 1 ? 'epoch logged' : 'epochs logged'}
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              {lang === 'ar'
                ? 'تتبع تفصيلي لمصفوفات W, b وتدرجات dW, db ومشتقات قاعدة السلسلة لحظياً'
                : 'Real-time trace of W, b matrices, gradients dW, db and chain-rule derivatives'}
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {/* Mode toggle */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-0.5 flex text-xs">
            <button
              onClick={() => setViewMode('terminal')}
              className={`px-2.5 py-1 rounded transition-colors ${
                viewMode === 'terminal'
                  ? 'bg-indigo-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {lang === 'ar' ? 'سجل الطرفية (Terminal)' : 'Terminal Text'}
            </button>
            <button
              onClick={() => setViewMode('visual_cards')}
              className={`px-2.5 py-1 rounded transition-colors ${
                viewMode === 'visual_cards'
                  ? 'bg-indigo-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {lang === 'ar' ? 'البطاقات الهيكلية (Visual)' : 'Visual Cards'}
            </button>
          </div>

          <button
            onClick={handleCopy}
            disabled={snapshots.length === 0}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed border border-slate-700/60"
            title={t.copyLogs}
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? (lang === 'ar' ? 'تم النسخ!' : 'Copied!') : t.copyLogs}</span>
          </button>

          <button
            onClick={handleDownloadPythonInspector}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white transition-colors shadow-sm"
            title="Download Python standalone script with Math Inspector"
          >
            <Download className="w-3.5 h-3.5" />
            <span>{lang === 'ar' ? 'تنزيل كود بايثون مع السجل' : 'Download Python Script'}</span>
          </button>

          <button
            onClick={onClearLogs}
            disabled={snapshots.length === 0}
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors border border-transparent hover:border-rose-500/20 disabled:opacity-30 disabled:cursor-not-allowed"
            title={t.clearLogs}
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Epoch Scrubber / Selector */}
      {snapshots.length > 0 && (
        <div className="bg-slate-950/80 px-4 py-2 border-b border-slate-800/80 flex items-center gap-2 overflow-x-auto text-xs no-scrollbar">
          <span className="text-slate-400 font-mono text-[11px] whitespace-nowrap">
            {lang === 'ar' ? 'الدورات المسجلة:' : 'Epochs:'}
          </span>
          <button
            onClick={() => setSelectedEpochIdx(null)}
            className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors whitespace-nowrap ${
              selectedEpochIdx === null
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900 border border-slate-800'
            }`}
          >
            {lang === 'ar' ? 'الأحدث (Live)' : 'Latest (Live)'}
          </button>
          {snapshots.map((snap: InspectorEpochLog, idx: number) => (
            <button
              key={idx}
              onClick={() => setSelectedEpochIdx(idx)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors whitespace-nowrap ${
                selectedEpochIdx === idx
                  ? 'bg-indigo-600 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200 bg-slate-900 border border-slate-800'
              }`}
            >
              #{snap.epoch} (L={snap.loss.toFixed(3)})
            </button>
          ))}
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto p-4 min-h-[350px]">
        {snapshots.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500">
            <Activity className="w-12 h-12 text-slate-600 stroke-1 mb-3 animate-pulse" />
            <h4 className="text-sm font-semibold text-slate-300 mb-1">
              {lang === 'ar' ? 'سجل التشريح الرياضي بانتظار التدريب' : 'Math Inspector awaiting execution'}
            </h4>
            <p className="text-xs text-slate-400 max-w-md">
              {lang === 'ar'
                ? 'اضغط على "بدء التدريب" أو "خطوة واحدة (Epoch)" في الشريط العلوي لتشغيل دورات التدريب ومشاهدة تشريح المصفوفات والتدرجات في الوقت الفعلي.'
                : 'Click "Train Model" or "Step (1 Epoch)" in the top control bar to stream live forward passes, chain-rule gradients, and Adam weight updates.'}
            </p>
          </div>
        ) : viewMode === 'terminal' ? (
          /* Pure Python Terminal Aesthetic */
          <div className="bg-black/90 font-mono text-[12px] text-emerald-400/90 p-4 rounded-lg border border-slate-800 leading-relaxed overflow-x-auto select-text shadow-inner" dir="ltr">
            <pre className="whitespace-pre font-mono">
              {rawTerminalText}
            </pre>
            <div ref={terminalEndRef} />
          </div>
        ) : (
          /* Visual Structured Cards Mode */
          activeSnapshot && (
            <div className="space-y-4" dir="ltr">
              {/* Epoch Metric Summary */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950 p-3 rounded-lg border border-slate-800">
                <div>
                  <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Epoch</div>
                  <div className="text-lg font-bold font-mono text-emerald-400">#{activeSnapshot.epoch}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Loss</div>
                  <div className="text-lg font-bold font-mono text-rose-400">{activeSnapshot.loss.toFixed(5)}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Samples Processed</div>
                  <div className="text-lg font-bold font-mono text-cyan-400">{activeSnapshot.targets.length}</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Optimizer</div>
                  <div className="text-lg font-bold font-mono text-indigo-400">Adam (lr={model.optimizer?.learning_rate || 0.01})</div>
                </div>
              </div>

              {/* Chain Rule Formula Banner */}
              <div className="p-3 bg-indigo-950/40 border border-indigo-800/40 rounded-lg text-xs font-mono text-indigo-200 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <span className="font-bold text-indigo-300">Chain Rule: </span>
                  <span className="text-amber-300">dW = Xᵀ • dZ</span> |{' '}
                  <span className="text-cyan-300">db = ∑ dZ</span> |{' '}
                  <span className="text-emerald-300">dX = dZ • Wᵀ</span>
                </div>
                <div className="text-[11px] text-slate-400">
                  dLoss / dZ = dLoss / da ⊙ σ&apos;(z)
                </div>
              </div>

              {/* Layer Cards */}
              <div className="space-y-3">
                {activeSnapshot.layers.map((layer: InspectorLayerLog, lIdx: number) => {
                  const inRows = layer.inputs?.length || 0;
                  const inCols = layer.inputs?.[0]?.length || 0;
                  const outRows = layer.output?.length || 0;
                  const outCols = layer.output?.[0]?.length || 0;
                  const dWNorm = getMatrixNorm(layer.dweights);
                  const dbNorm = getMatrixNorm(layer.dbiases);
                  const dXNorm = getMatrixNorm(layer.dinputs);

                  return (
                    <div
                      key={lIdx}
                      className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3 hover:border-slate-700 transition-all"
                    >
                      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80 mb-2">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-xs font-bold">
                            Layer {lIdx + 1}: {layer.layerName}
                          </span>
                          <span className="text-xs text-slate-400 font-mono">{layer.layerType}</span>
                        </div>
                        <div className="text-xs font-mono text-slate-400">
                          Input: <span className="text-slate-200">[{inRows} × {inCols}]</span>{' '}
                          <ArrowRight className="w-3 h-3 inline text-slate-500 mx-1" />
                          Output: <span className="text-slate-200">[{outRows} × {outCols}]</span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
                        {/* Forward Pass Stats */}
                        <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800/60">
                          <div className="text-emerald-400 font-semibold mb-1 flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                            [1] Forward Pass
                          </div>
                          <div className="text-slate-300 text-[11px] space-y-0.5">
                            <div>Shape: [{outRows} × {outCols}]</div>
                            {layer.weights && (
                              <div>Weights Shape: [{layer.weights.length} × {layer.weights[0]?.length || 0}]</div>
                            )}
                            {layer.biases && (
                              <div>Biases Shape: [1 × {layer.biases[0]?.length || 0}]</div>
                            )}
                          </div>
                        </div>

                        {/* Backpropagation Gradients */}
                        <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800/60">
                          <div className="text-amber-400 font-semibold mb-1 flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                            [2] Backpropagation & Gradients
                          </div>
                          <div className="text-slate-300 text-[11px] space-y-0.5">
                            {layer.dweights ? (
                              <>
                                <div>||dW||₂ (Weight Gradient Norm): <span className="text-amber-300 font-bold">{dWNorm.toFixed(5)}</span></div>
                                <div>||db||₂ (Bias Gradient Norm): <span className="text-amber-300 font-bold">{dbNorm.toFixed(5)}</span></div>
                                {layer.dinputs && (
                                  <div>||dX||₂ (Propagated upstream): <span className="text-cyan-300 font-bold">{dXNorm.toFixed(5)}</span></div>
                                )}
                              </>
                            ) : (
                              <div className="text-slate-400 italic">Activation pass-through: dZ = dOut ⊙ σ&apos;(z)</div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )
        )}
      </div>
    </div>
  );
};
