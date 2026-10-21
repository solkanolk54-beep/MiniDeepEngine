import React, { useRef, useEffect, useState } from 'react';
import { TrendingDown, Activity } from 'lucide-react';
import { Language, translations } from '../translations';

interface LossChartProps {
  lossHistory: number[];
  metricHistory: number[];
  lang: Language;
}

export const LossChart: React.FC<LossChartProps> = ({
  lossHistory,
  metricHistory,
  lang,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const t = translations[lang];
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Padding
    const pTop = 20;
    const pBottom = 30;
    const pLeft = 45;
    const pRight = 20;
    const plotW = width - pLeft - pRight;
    const plotH = height - pTop - pBottom;

    if (lossHistory.length < 2) {
      // Empty state
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Cairo, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(lang === 'ar' ? 'سيبدأ رسم منحنى الخسارة عند بدء التدريب...' : 'Loss curve will appear once training starts...', width / 2, height / 2);
      return;
    }

    // Subsample points if history is large
    const maxPoints = 300;
    const step = Math.max(1, Math.floor(lossHistory.length / maxPoints));
    const sampledLoss: { epoch: number; val: number }[] = [];
    const sampledMetric: { epoch: number; val: number }[] = [];

    for (let i = 0; i < lossHistory.length; i += step) {
      sampledLoss.push({ epoch: i, val: lossHistory[i] });
      if (metricHistory[i] !== undefined) {
        sampledMetric.push({ epoch: i, val: metricHistory[i] });
      }
    }
    // Always include latest
    if (sampledLoss[sampledLoss.length - 1].epoch !== lossHistory.length - 1) {
      sampledLoss.push({ epoch: lossHistory.length - 1, val: lossHistory[lossHistory.length - 1] });
      if (metricHistory[lossHistory.length - 1] !== undefined) {
        sampledMetric.push({ epoch: lossHistory.length - 1, val: metricHistory[lossHistory.length - 1] });
      }
    }

    // Min and max loss (with logarithmic or clamped scale)
    const minLoss = Math.min(...sampledLoss.map(d => d.val));
    const maxLoss = Math.max(...sampledLoss.map(d => d.val), 0.1);

    // Grid lines & Y-axis labels
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 4; i++) {
      const y = pTop + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pLeft, y);
      ctx.lineTo(width - pRight, y);
      ctx.stroke();

      const val = maxLoss - (i / 4) * (maxLoss - minLoss);
      ctx.fillText(val.toFixed(3), pLeft - 6, y + 3);
    }

    // Epoch ticks along bottom
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
      const x = pLeft + (plotW / 4) * i;
      const ep = Math.round((i / 4) * (lossHistory.length - 1));
      ctx.fillText(ep.toString(), x, height - 10);
    }

    // Draw Loss Path
    ctx.beginPath();
    ctx.strokeStyle = '#38bdf8'; // Cyan
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
    ctx.shadowBlur = 6;

    for (let i = 0; i < sampledLoss.length; i++) {
      const x = pLeft + (i / (sampledLoss.length - 1)) * plotW;
      const norm = (sampledLoss[i].val - minLoss) / ((maxLoss - minLoss) || 1);
      const y = pTop + plotH - norm * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw Metric Path (Accuracy: green) if present
    if (sampledMetric.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = '#10b981'; // Emerald
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);

      for (let i = 0; i < sampledMetric.length; i++) {
        const x = pLeft + (i / (sampledMetric.length - 1)) * plotW;
        // Metric is usually 0..1 (Accuracy)
        const y = pTop + plotH - Math.min(1, Math.max(0, sampledMetric[i].val)) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [lossHistory, metricHistory, lang]);

  const minLossVal = lossHistory.length ? Math.min(...lossHistory) : 0;
  const currentLossVal = lossHistory.length ? lossHistory[lossHistory.length - 1] : 0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col shadow-xl">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <TrendingDown className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-white">{t.lossCurve}</h2>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block shadow-sm" />
            <span className="text-slate-300 font-mono text-[11px]">
              Loss: {currentLossVal.toFixed(4)}
            </span>
          </div>

          {metricHistory.length > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block shadow-sm" />
              <span className="text-slate-300 font-mono text-[11px]">
                Metric: {(metricHistory[metricHistory.length - 1] * 100).toFixed(1)}%
              </span>
            </div>
          )}

          {lossHistory.length > 0 && (
            <span className="bg-indigo-950/80 text-indigo-300 border border-indigo-700/50 text-[10px] px-2 py-0.5 rounded-full font-mono">
              Min: {minLossVal.toFixed(5)}
            </span>
          )}
        </div>
      </div>

      <div className="relative mt-3 rounded-xl overflow-hidden border border-slate-700/60 bg-slate-950 flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={380}
          height={320}
          className="w-full max-w-full h-auto select-none"
        />
      </div>
    </div>
  );
};
