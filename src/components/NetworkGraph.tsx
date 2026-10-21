import React from 'react';
import { NeuralNetwork } from '../engine/model';
import { DenseLayer } from '../engine/layers';
import { Language, translations } from '../translations';
import { Network, MousePointerClick } from 'lucide-react';

interface NetworkGraphProps {
  model: NeuralNetwork;
  lang: Language;
  onSelectNeuron: (layerIdx: number, neuronIdx: number) => void;
  selectedNeuron: { layerIdx: number; neuronIdx: number } | null;
}

export const NetworkGraph: React.FC<NetworkGraphProps> = ({
  model,
  lang,
  onSelectNeuron,
  selectedNeuron,
}) => {
  const t = translations[lang];

  // Extract dense layers to visualize architecture
  const denseLayers = model.layers.filter(l => l instanceof DenseLayer) as DenseLayer[];

  if (denseLayers.length === 0) {
    return null;
  }

  // Derive layer structure: [input_dim, hidden1_dim, ..., output_dim]
  const layerSizes: number[] = [denseLayers[0].n_inputs];
  denseLayers.forEach(l => layerSizes.push(l.n_neurons));

  // Cap visualization count per layer for clean UI if layer has many neurons
  const MAX_DISPLAY_PER_LAYER = 8;

  const svgWidth = 720;
  const svgHeight = 280;
  const numLayers = layerSizes.length;
  const layerSpacing = (svgWidth - 100) / (numLayers - 1 || 1);

  // Compute node coordinates
  const nodePositions: { x: number; y: number; layerIdx: number; neuronIdx: number; isTruncated?: boolean }[][] = [];

  for (let l = 0; l < numLayers; l++) {
    const totalCount = layerSizes[l];
    const displayCount = Math.min(totalCount, MAX_DISPLAY_PER_LAYER);
    const x = 50 + l * layerSpacing;
    const layerNodes = [];

    const spacing = (svgHeight - 60) / (displayCount + 1);

    for (let i = 0; i < displayCount; i++) {
      const y = 30 + (i + 1) * spacing;
      layerNodes.push({
        x,
        y,
        layerIdx: l,
        neuronIdx: i,
        isTruncated: totalCount > MAX_DISPLAY_PER_LAYER && i === Math.floor(displayCount / 2),
      });
    }
    nodePositions.push(layerNodes);
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col shadow-xl">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800 gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-indigo-400" />
          <h2 className="text-sm font-bold text-white">{t.networkArchitecture}</h2>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <MousePointerClick className="w-3.5 h-3.5 text-indigo-400" />
          <span className="text-[11px]">{t.clickToInspect}</span>
        </div>
      </div>

      <div className="relative mt-3 w-full overflow-x-auto rounded-xl border border-slate-700/60 bg-slate-950 p-2 flex justify-center">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full max-w-3xl h-auto min-w-[500px]"
        >
          <defs>
            {/* Gradients */}
            <radialGradient id="nodeActiveGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#818cf8" />
              <stop offset="100%" stopColor="#4338ca" />
            </radialGradient>
            <radialGradient id="nodeSelectGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#0284c7" />
            </radialGradient>
          </defs>

          {/* Draw Synapses (Weights) between consecutive layers */}
          <g key="synapses-group">
            {denseLayers.flatMap((dense, lIdx) => {
              const currentNodes = nodePositions[lIdx];
              const nextNodes = nodePositions[lIdx + 1];
              if (!currentNodes || !nextNodes) return [];

              return currentNodes.flatMap((srcNode, sIdx) => {
                return nextNodes.map((dstNode, dIdx) => {
                  // Get weight if exists
                  const w = dense.weights[sIdx]?.[dIdx] ?? 0;
                  const isPositive = w >= 0;
                  const absW = Math.abs(w);
                  const strokeWidth = Math.min(4, Math.max(0.6, absW * 2.5));
                  const strokeColor = isPositive
                    ? `rgba(56, 189, 248, ${Math.min(0.85, 0.2 + absW)})`
                    : `rgba(244, 63, 94, ${Math.min(0.85, 0.2 + absW)})`;

                  return (
                    <line
                      key={`syn-${lIdx}-${sIdx}-${dIdx}`}
                      x1={srcNode.x}
                      y1={srcNode.y}
                      x2={dstNode.x}
                      y2={dstNode.y}
                      stroke={strokeColor}
                      strokeWidth={strokeWidth}
                      strokeLinecap="round"
                      className="transition-all duration-300 hover:stroke-amber-400 hover:stroke-[3px] cursor-pointer"
                    />
                  );
                });
              });
            })}
          </g>

          {/* Draw Neurons */}
          {nodePositions.map((layerNodes, lIdx) => {
            const layerTypeLabel =
              lIdx === 0
                ? lang === 'ar' ? 'المدخلات (Inputs)' : 'Inputs'
                : lIdx === nodePositions.length - 1
                ? lang === 'ar' ? 'المخرجات (Output)' : 'Output'
                : `${lang === 'ar' ? 'طبقة مخفية' : 'Hidden'} ${lIdx}`;

            return (
              <g key={`layer-${lIdx}`}>
                {/* Layer header tag */}
                <text
                  x={layerNodes[0]?.x || 0}
                  y={18}
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  {layerTypeLabel}
                </text>

                {layerNodes.map((node) => {
                  const isSelected =
                    selectedNeuron?.layerIdx === node.layerIdx &&
                    selectedNeuron?.neuronIdx === node.neuronIdx;

                  // Extract neuron activation output if available
                  let activationVal: number | null = null;
                  if (lIdx === 0) {
                    activationVal = null;
                  } else {
                    const dense = denseLayers[lIdx - 1];
                    if (dense?.output?.[0]?.[node.neuronIdx] !== undefined) {
                      activationVal = dense.output[0][node.neuronIdx];
                    }
                  }

                  return (
                    <g
                      key={`node-${node.layerIdx}-${node.neuronIdx}`}
                      role="button"
                      tabIndex={0}
                      aria-label={lIdx === 0 ? `Input Neuron x${node.neuronIdx + 1}` : `Neuron L${node.layerIdx} N${node.neuronIdx + 1}`}
                      onClick={() => onSelectNeuron(node.layerIdx, node.neuronIdx)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          onSelectNeuron(node.layerIdx, node.neuronIdx);
                        }
                      }}
                      className="cursor-pointer group outline-none"
                    >
                      {/* Glow circle on select */}
                      {isSelected && (
                        <circle
                          cx={node.x}
                          cy={node.y}
                          r={18}
                          fill="none"
                          stroke="#38bdf8"
                          strokeWidth={2}
                          strokeDasharray="4 2"
                          className="animate-spin"
                        />
                      )}

                      {/* Main Neuron Circle */}
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={13}
                        fill={isSelected ? 'url(#nodeSelectGrad)' : 'url(#nodeActiveGrad)'}
                        stroke="#ffffff"
                        strokeWidth={isSelected ? 2 : 1}
                        className="transition-transform group-hover:scale-110"
                      />

                      {/* Neuron Index / Label */}
                      <text
                        x={node.x}
                        y={node.y + 3.5}
                        fill="#ffffff"
                        fontSize="9"
                        fontWeight="bold"
                        fontFamily="JetBrains Mono, monospace"
                        textAnchor="middle"
                        pointerEvents="none"
                      >
                        {lIdx === 0 ? `x${node.neuronIdx + 1}` : `n${node.neuronIdx + 1}`}
                      </text>

                      {/* Small value badge below neuron */}
                      {activationVal !== null && (
                        <text
                          x={node.x}
                          y={node.y + 22}
                          fill="#38bdf8"
                          fontSize="8"
                          fontFamily="JetBrains Mono, monospace"
                          textAnchor="middle"
                        >
                          {activationVal.toFixed(2)}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
