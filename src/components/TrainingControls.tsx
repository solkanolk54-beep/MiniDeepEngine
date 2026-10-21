import React from 'react';
import { Play, Pause, StepForward, RotateCcw, Zap, Gauge, CheckCircle2 } from 'lucide-react';
import { Language, translations } from '../translations';
import { EvaluationMetrics } from '../engine/model';

interface TrainingControlsProps {
  lang: Language;
  isTraining: boolean;
  onToggleTraining: () => void;
  onStep: () => void;
  onReset: () => void;
  epochs: number;
  currentLoss: number;
  metrics: EvaluationMetrics | null;
  learningRate: number;
  speed: number;
  setSpeed: (s: number) => void;
  isConverged?: boolean;
}

export const TrainingControls: React.FC<TrainingControlsProps> = ({
  lang,
  isTraining,
  onToggleTraining,
  onStep,
  onReset,
  epochs,
  currentLoss,
  metrics,
  learningRate,
  speed,
  setSpeed,
  isConverged,
}) => {
  const t = translations[lang];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={onToggleTraining}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm shadow-lg transition active:scale-95 ${
              isTraining
                ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-600/25 ring-2 ring-amber-400/30'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30 ring-2 ring-indigo-400/30'
            }`}
          >
            {isTraining ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
            <span>{isTraining ? t.pause : t.train}</span>
          </button>

          <button
            onClick={onStep}
            disabled={isTraining}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl font-medium text-sm bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 border border-slate-700 transition"
          >
            <StepForward className="w-4 h-4" />
            <span>{t.step}</span>
          </button>

          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-3.5 py-2.5 rounded-xl font-medium text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
            title={t.reset}
          >
            <RotateCcw className="w-4 h-4" />
            <span className="hidden sm:inline">{t.reset}</span>
          </button>

          {/* Speed Selector */}
          <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded-xl p-1 text-xs">
            <Gauge className="w-3.5 h-3.5 text-slate-400 ml-1.5 mr-1" />
            {[1, 5, 20, 50].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2.5 py-1 rounded-lg font-mono font-medium transition ${
                  speed === s
                    ? 'bg-indigo-600 text-white font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Live Metrics HUD */}
        <div className="flex items-center gap-3 sm:gap-6 flex-wrap">
          {/* Epoch */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{t.epochs}</span>
            <span className="font-mono text-lg font-bold text-white">{epochs.toLocaleString()}</span>
          </div>

          {/* Loss */}
          <div className="flex flex-col">
            <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{t.loss}</span>
            <span className="font-mono text-lg font-bold text-cyan-400">
              {currentLoss.toFixed(5)}
            </span>
          </div>

          {/* Metric (Accuracy or RMSE) */}
          {metrics?.accuracy !== undefined && (
            <div className="flex flex-col">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{t.accuracy}</span>
              <span className="font-mono text-lg font-bold text-emerald-400 flex items-center gap-1">
                {(metrics.accuracy * 100).toFixed(1)}%
                {metrics.accuracy >= 0.99 && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              </span>
            </div>
          )}

          {metrics?.rmse !== undefined && (
            <div className="flex flex-col">
              <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{t.rmse}</span>
              <span className="font-mono text-lg font-bold text-emerald-400">
                {metrics.rmse.toFixed(4)}
              </span>
            </div>
          )}

          {/* Learning Rate */}
          <div className="flex flex-col hidden md:flex">
            <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">{t.learningRate}</span>
            <span className="font-mono text-base font-semibold text-slate-300">
              {learningRate}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
