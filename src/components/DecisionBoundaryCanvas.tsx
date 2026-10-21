import React, { useRef, useEffect, useState } from 'react';
import { NeuralNetwork } from '../engine/model';
import { Dataset } from '../engine/datasets';
import { Language, translations } from '../translations';
import { PlusCircle, Info, Crosshair } from 'lucide-react';

interface DecisionBoundaryCanvasProps {
  model: NeuralNetwork;
  dataset: Dataset;
  lang: Language;
  onAddPoint: (x1: number, x2: number, label: number) => void;
  epoch: number;
}

export const DecisionBoundaryCanvas: React.FC<DecisionBoundaryCanvasProps> = ({
  model,
  dataset,
  lang,
  onAddPoint,
  epoch,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const t = translations[lang];
  const [newPointClass, setNewPointClass] = useState<number>(1);
  const [hoverInfo, setHoverInfo] = useState<{ x1: number; x2: number; pred: number } | null>(null);

  const { xMin, xMax, yMin, yMax } = dataset.domain;

  // Render decision boundary whenever model epoch changes or dataset changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Grid sampling resolution
    const res = 50;
    const cellW = width / res;
    const cellH = height / res;

    // Create batch input for all grid cells
    const gridInputs: number[][] = [];
    for (let j = 0; j < res; j++) {
      const yVal = yMax - (j / (res - 1)) * (yMax - yMin);
      for (let i = 0; i < res; i++) {
        const xVal = xMin + (i / (res - 1)) * (xMax - xMin);
        gridInputs.push([xVal, yVal]);
      }
    }

    // Run forward pass on batch
    let predictions: number[][] = [];
    try {
      predictions = model.forward(gridInputs, false);
    } catch {
      predictions = gridInputs.map(() => [0.5]);
    }

    // Render background heatmap
    let idx = 0;
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const pred = predictions[idx] ? predictions[idx][0] : 0.5;
        idx++;

        if (dataset.type === 'classification') {
          // Blue for 1, Purple/Red for 0
          const p = Math.max(0, Math.min(1, pred));
          const r = Math.floor(180 * (1 - p) + 30 * p);
          const g = Math.floor(40 * (1 - p) + 120 * p);
          const b = Math.floor(70 * (1 - p) + 240 * p);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        } else {
          // Continuous heat palette for regression
          const normalized = Math.max(0, Math.min(1, (pred - yMin) / ((yMax - yMin) || 1)));
          const r = Math.floor(255 * normalized);
          const g = Math.floor(180 * (1 - Math.abs(normalized - 0.5) * 2));
          const b = Math.floor(255 * (1 - normalized));
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        }

        ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
      }
    }

    // Draw coordinate axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;

    // Zero axes if within domain
    if (xMin < 0 && xMax > 0) {
      const zeroX = ((0 - xMin) / (xMax - xMin)) * width;
      ctx.beginPath();
      ctx.moveTo(zeroX, 0);
      ctx.lineTo(zeroX, height);
      ctx.stroke();
    }
    if (yMin < 0 && yMax > 0) {
      const zeroY = ((yMax - 0) / (yMax - yMin)) * height;
      ctx.beginPath();
      ctx.moveTo(0, zeroY);
      ctx.lineTo(width, zeroY);
      ctx.stroke();
    }

    // Draw dataset points
    for (let k = 0; k < dataset.X.length; k++) {
      const pt = dataset.X[k];
      const target = dataset.y[k][0];
      const px = ((pt[0] - xMin) / (xMax - xMin)) * width;
      const py = ((yMax - pt[1]) / (yMax - yMin)) * height;

      // Outer glow & circle
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, 2 * Math.PI);

      if (dataset.type === 'classification') {
        const isPos = target >= 0.5;
        ctx.fillStyle = isPos ? '#38bdf8' : '#f43f5e';
        ctx.shadowColor = isPos ? 'rgba(56, 189, 248, 0.8)' : 'rgba(244, 63, 94, 0.8)';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      } else {
        ctx.fillStyle = '#f59e0b';
        ctx.shadowColor = 'rgba(245, 158, 11, 0.8)';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
    }
  }, [model, dataset, epoch, xMin, xMax, yMin, yMax]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const xVal = xMin + (clickX / canvas.width) * (xMax - xMin);
    const yVal = yMax - (clickY / canvas.height) * (yMax - yMin);

    onAddPoint(Number(xVal.toFixed(3)), Number(yVal.toFixed(3)), newPointClass);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const xVal = xMin + (mouseX / canvas.width) * (xMax - xMin);
    const yVal = yMax - (mouseY / canvas.height) * (yMax - yMin);

    try {
      const pred = model.forward([[xVal, yVal]], false);
      setHoverInfo({
        x1: Number(xVal.toFixed(2)),
        x2: Number(yVal.toFixed(2)),
        pred: Number((pred[0]?.[0] || 0).toFixed(3)),
      });
    } catch {
      setHoverInfo(null);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col shadow-xl">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <Crosshair className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-white">{t.decisionBoundary}</h2>
        </div>

        {/* Legend & Add Point Selector */}
        {dataset.type === 'classification' && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-400 text-[11px] hidden sm:inline">{lang === 'ar' ? 'إضافة نقطة:' : 'Add point:'}</span>
            <button
              onClick={() => setNewPointClass(1)}
              className={`flex items-center gap-1 px-2 py-0.5 rounded-md border transition ${
                newPointClass === 1
                  ? 'bg-cyan-950 text-cyan-300 border-cyan-500 font-bold ring-1 ring-cyan-500/50'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" />
              <span>{t.positiveClass}</span>
            </button>
            <button
              onClick={() => setNewPointClass(0)}
              className={`flex items-center gap-1 px-2 py-0.5 rounded-md border transition ${
                newPointClass === 0
                  ? 'bg-rose-950 text-rose-300 border-rose-500 font-bold ring-1 ring-rose-500/50'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" />
              <span>{t.negativeClass}</span>
            </button>
          </div>
        )}
      </div>

      {/* Canvas Area */}
      <div className="relative mt-3 rounded-xl overflow-hidden border border-slate-700/60 bg-slate-950 flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={380}
          height={320}
          onClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverInfo(null)}
          className="w-full max-w-full h-auto cursor-crosshair select-none"
        />

        {/* Live Hover Coordinate Readout */}
        {hoverInfo && (
          <div className="absolute top-2 left-2 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-lg px-2.5 py-1 text-xs font-mono text-slate-200 pointer-events-none shadow-lg">
            x₁: <span className="text-cyan-400 font-bold">{hoverInfo.x1}</span>, x₂:{' '}
            <span className="text-cyan-400 font-bold">{hoverInfo.x2}</span> | ŷ:{' '}
            <span className="text-emerald-400 font-bold">{hoverInfo.pred}</span>
          </div>
        )}

        <div className="absolute bottom-2 right-2 bg-slate-950/80 backdrop-blur rounded px-2 py-0.5 text-[10px] text-slate-400 pointer-events-none">
          {t.clickToAddPoint}
        </div>
      </div>
    </div>
  );
};
