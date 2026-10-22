/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { TrainingControls } from './components/TrainingControls';
import { DecisionBoundaryCanvas } from './components/DecisionBoundaryCanvas';
import { LossChart } from './components/LossChart';
import { NetworkGraph } from './components/NetworkGraph';
import { NeuronInspectorModal } from './components/NeuronInspectorModal';
import { MathStepInspector } from './components/MathStepInspector';
import { ArchitectureBuilder } from './components/ArchitectureBuilder';
import { CodeExportModal } from './components/CodeExportModal';
import { MathGuideModal } from './components/MathGuideModal';
import { MathInspectorLogger } from './components/MathInspectorLogger';
import { CustomActivationLab } from './components/CustomActivationLab';
import { XORBenchmarkLab } from './components/XORBenchmarkLab';
import { MathInspectorLogger as EngineLogger } from './engine/inspectorLogger';
import { NeuralNetwork, EvaluationMetrics } from './engine/model';
import { DenseLayer } from './engine/layers';
import { Activation_ReLU, Activation_Sigmoid, LayerInterface } from './engine/activations';
import { Loss_MSE, LossInterface } from './engine/loss';
import { Optimizer_Adam, OptimizerInterface } from './engine/optimizers';
import { Dataset, DATASET_PRESETS, generateXOR } from './engine/datasets';
import { Language, translations } from './translations';
import { PlaySquare, Compass, SlidersHorizontal, Sparkles, Terminal, Cpu, Flame } from 'lucide-react';

export default function App() {
  const [lang, setLang] = useState<Language>('ar');
  const t = translations[lang];

  // Active dataset
  const [dataset, setDataset] = useState<Dataset>(DATASET_PRESETS[0]); // XOR default

  // Math Inspector Logger
  const [inspectorLogger] = useState(() => new EngineLogger(150));
  const [, setLogTick] = useState(0);

  // Model instance
  const [model, setModel] = useState<NeuralNetwork>(() => {
    const net = new NeuralNetwork();
    net.add(new DenseLayer(2, 8));
    net.add(new Activation_ReLU());
    net.add(new DenseLayer(8, 1));
    net.add(new Activation_Sigmoid());
    net.set(new Loss_MSE(), new Optimizer_Adam(0.04));
    net.setLogger(new EngineLogger(150));
    return net;
  });

  // Ensure logger is attached
  useEffect(() => {
    model.setLogger(inspectorLogger);
  }, [model, inspectorLogger]);

  // Training state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epochs, setEpochs] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [metricHistory, setMetricHistory] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [speed, setSpeed] = useState<number>(5);

  // Active Tab & SubTab
  const [activeTab, setActiveTab] = useState<'playground' | 'math' | 'architecture' | 'rdlab'>('playground');
  const [rdSubTab, setRdSubTab] = useState<'xor' | 'logger' | 'activation'>('xor');

  // Modals & Inspection
  const [selectedNeuron, setSelectedNeuron] = useState<{ layerIdx: number; neuronIdx: number } | null>(null);
  const [showCodeModal, setShowCodeModal] = useState<boolean>(false);
  const [showGuideModal, setShowGuideModal] = useState<boolean>(false);

  // Sync HTML dir attribute for RTL/LTR
  useEffect(() => {
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.lang = lang;
  }, [lang]);

  // Reset model weights
  const handleReset = useCallback(() => {
    setIsTraining(false);
    setEpochs(0);
    setLossHistory([]);
    setMetricHistory([]);

    const net = new NeuralNetwork();
    const inDim = dataset.X[0]?.length || 2;
    const outDim = dataset.y[0]?.length || 1;

    if (dataset.type === 'classification') {
      net.add(new DenseLayer(inDim, 8));
      net.add(new Activation_ReLU());
      net.add(new DenseLayer(8, outDim));
      net.add(new Activation_Sigmoid());
      net.set(new Loss_MSE(), new Optimizer_Adam(0.04));
    } else {
      net.add(new DenseLayer(inDim, 8));
      net.add(new Activation_ReLU());
      net.add(new DenseLayer(8, outDim));
      net.set(new Loss_MSE(), new Optimizer_Adam(0.02));
    }

    const initialMetrics = net.evaluate(dataset.X, dataset.y);
    net.setLogger(inspectorLogger);
    setCurrentLoss(initialMetrics.loss);
    setMetrics(initialMetrics);
    setModel(net);
  }, [dataset, inspectorLogger]);

  // Switch dataset
  const handleSelectDataset = (d: Dataset) => {
    setDataset(d);
    setIsTraining(false);
    setEpochs(0);
    setLossHistory([]);
    setMetricHistory([]);

    const net = new NeuralNetwork();
    const inDim = d.X[0]?.length || 2;
    const outDim = d.y[0]?.length || 1;

    if (d.type === 'classification') {
      net.add(new DenseLayer(inDim, 8));
      net.add(new Activation_ReLU());
      net.add(new DenseLayer(8, outDim));
      net.add(new Activation_Sigmoid());
      net.set(new Loss_MSE(), new Optimizer_Adam(0.04));
    } else {
      net.add(new DenseLayer(inDim, 8));
      net.add(new Activation_ReLU());
      net.add(new DenseLayer(8, outDim));
      net.set(new Loss_MSE(), new Optimizer_Adam(0.02));
    }

    const initMetrics = net.evaluate(d.X, d.y);
    net.setLogger(inspectorLogger);
    setCurrentLoss(initMetrics.loss);
    setMetrics(initMetrics);
    setModel(net);
  };

  // Perform a single step
  const handleStep = useCallback(() => {
    if (!model || dataset.X.length === 0) return;
    const res = model.step(dataset.X, dataset.y);
    setEpochs(prev => prev + 1);
    setCurrentLoss(res.loss);
    setMetrics(res.metrics);
    setLossHistory([...model.loss_history]);
    setMetricHistory([...model.metric_history]);
    setLogTick(t => t + 1);
  }, [model, dataset]);

  // Animation frame loop for continuous training
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isTraining) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      return;
    }

    let isRunning = true;

    const loop = () => {
      if (!isRunning) return;

      // Execute `speed` iterations per frame
      for (let s = 0; s < speed; s++) {
        model.step(dataset.X, dataset.y);
      }

      const latestLoss = model.loss_history[model.loss_history.length - 1] || 0;
      const latestMetrics = model.evaluate(dataset.X, dataset.y);

      setEpochs(prev => prev + speed);
      setCurrentLoss(latestLoss);
      setMetrics(latestMetrics);
      setLossHistory([...model.loss_history]);
      setMetricHistory([...model.metric_history]);
      setLogTick(t => t + 1);

      animationRef.current = requestAnimationFrame(loop);
    };

    animationRef.current = requestAnimationFrame(loop);

    return () => {
      isRunning = false;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isTraining, model, dataset, speed]);

  // Add custom point on canvas
  const handleAddPoint = (x1: number, x2: number, label: number) => {
    setDataset(prev => ({
      ...prev,
      X: [...prev.X, [x1, x2]],
      y: [...prev.y, [label]],
    }));
  };

  // Apply custom architecture from Builder
  const handleApplyArchitecture = (
    layers: LayerInterface[],
    loss: LossInterface,
    optimizer: OptimizerInterface
  ) => {
    setIsTraining(false);
    setEpochs(0);
    setLossHistory([]);
    setMetricHistory([]);

    const net = new NeuralNetwork();
    layers.forEach(l => net.add(l));
    net.set(loss, optimizer);
    net.setLogger(inspectorLogger);

    const initMetrics = net.evaluate(dataset.X, dataset.y);
    setCurrentLoss(initMetrics.loss);
    setMetrics(initMetrics);
    setModel(net);
    setActiveTab('playground');
  };

  // Import model from JSON
  const handleImportModel = (jsonStr: string) => {
    try {
      const net = NeuralNetwork.fromJSON(jsonStr);
      net.setLogger(inspectorLogger);
      setIsTraining(false);
      setEpochs(0);
      setLossHistory([]);
      setMetricHistory([]);
      const initMetrics = net.evaluate(dataset.X, dataset.y);
      setCurrentLoss(initMetrics.loss);
      setMetrics(initMetrics);
      setModel(net);
    } catch (err: any) {
      alert(`Import error: ${err.message}`);
    }
  };

  const handleClearLogs = () => {
    inspectorLogger.clear();
    setLogTick(t => t + 1);
  };

  const handleActivationInjected = () => {
    setIsTraining(false);
    const m = model.evaluate(dataset.X, dataset.y);
    setCurrentLoss(m.loss);
    setMetrics(m);
    // clone to trigger state update across components
    setModel(Object.assign(Object.create(Object.getPrototypeOf(model)), model));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Header */}
      <Header
        lang={lang}
        setLang={setLang}
        activeDataset={dataset}
        onSelectDataset={handleSelectDataset}
        onOpenGuide={() => setShowGuideModal(true)}
        onOpenCode={() => setShowCodeModal(true)}
        onReset={handleReset}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
          <button
            onClick={() => setActiveTab('playground')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs sm:text-sm transition ${
              activeTab === 'playground'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <PlaySquare className="w-4 h-4" />
            <span>{t.playgroundTab}</span>
          </button>

          <button
            onClick={() => setActiveTab('math')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs sm:text-sm transition ${
              activeTab === 'math'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Compass className="w-4 h-4" />
            <span>{t.mathInspectorTab}</span>
          </button>

          <button
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs sm:text-sm transition ${
              activeTab === 'architecture'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <SlidersHorizontal className="w-4 h-4" />
            <span>{t.architectureTab}</span>
          </button>

          <button
            onClick={() => setActiveTab('rdlab')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs sm:text-sm transition ${
              activeTab === 'rdlab'
                ? 'bg-gradient-to-r from-emerald-600 to-cyan-600 text-white shadow-lg shadow-emerald-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <Sparkles className="w-4 h-4 text-emerald-300" />
            <span>{t.rdLabTab}</span>
          </button>
        </div>

        {/* Tab 1: Playground */}
        {activeTab === 'playground' && (
          <div className="space-y-6">
            {/* Top Training HUD Controls */}
            <TrainingControls
              lang={lang}
              isTraining={isTraining}
              onToggleTraining={() => setIsTraining(!isTraining)}
              onStep={handleStep}
              onReset={handleReset}
              epochs={epochs}
              currentLoss={currentLoss}
              metrics={metrics}
              learningRate={model.optimizer?.learning_rate || 0.01}
              speed={speed}
              setSpeed={setSpeed}
              isConverged={metrics?.accuracy !== undefined && metrics.accuracy >= 0.99}
            />

            {/* Split Screen: Decision Boundary & Loss Curve */}
            <div className="grid md:grid-cols-2 gap-6">
              <DecisionBoundaryCanvas
                model={model}
                dataset={dataset}
                lang={lang}
                onAddPoint={handleAddPoint}
                epoch={epochs}
              />
              <LossChart
                lossHistory={lossHistory}
                metricHistory={metricHistory}
                lang={lang}
              />
            </div>

            {/* Live Interactive Computation Graph */}
            <NetworkGraph
              model={model}
              lang={lang}
              onSelectNeuron={(lIdx, nIdx) => setSelectedNeuron({ layerIdx: lIdx, neuronIdx: nIdx })}
              selectedNeuron={selectedNeuron}
            />
          </div>
        )}

        {/* Tab 2: Math Step Inspector */}
        {activeTab === 'math' && (
          <MathStepInspector
            model={model}
            lang={lang}
            onExecuteSingleStep={handleStep}
          />
        )}

        {/* Tab 3: Architecture Builder */}
        {activeTab === 'architecture' && (
          <ArchitectureBuilder
            inputDim={dataset.X[0]?.length || 2}
            outputDim={dataset.y[0]?.length || 1}
            onApplyArchitecture={handleApplyArchitecture}
            lang={lang}
          />
        )}

        {/* Tab 4: R&D Lab & Math Inspector Logger */}
        {activeTab === 'rdlab' && (
          <div className="space-y-6">
            {/* R&D Sub-Navigation Bar */}
            <div className="bg-slate-900/90 border border-slate-800 p-1.5 rounded-2xl flex flex-wrap items-center justify-between gap-2 shadow-xl">
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  onClick={() => setRdSubTab('xor')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-xs sm:text-sm transition ${
                    rdSubTab === 'xor'
                      ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-600/30'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/80'
                  }`}
                >
                  <Cpu className="w-4 h-4 text-emerald-300" />
                  <span>معضلة XOR والتصنيف غير الخطي (XOR Benchmark)</span>
                  <span className="text-[10px] bg-emerald-400/20 text-emerald-200 px-1.5 py-0.2 rounded font-mono">
                    المعيار الذهبي
                  </span>
                </button>

                <button
                  onClick={() => setRdSubTab('logger')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-xs sm:text-sm transition ${
                    rdSubTab === 'logger'
                      ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/80'
                  }`}
                >
                  <Terminal className="w-4 h-4 text-indigo-300" />
                  <span>سِجِل التشريح الرياضي الحي (Math Logger)</span>
                </button>

                <button
                  onClick={() => setRdSubTab('activation')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-xs sm:text-sm transition ${
                    rdSubTab === 'activation'
                      ? 'bg-amber-600 text-white shadow-lg shadow-amber-600/30'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800/80'
                  }`}
                >
                  <Sparkles className="w-4 h-4 text-amber-300" />
                  <span>مختبر دوال التفعيل المخصصة (Custom Activation)</span>
                </button>
              </div>
            </div>

            {/* Sub-view 1: XOR Benchmark Lab */}
            {rdSubTab === 'xor' && (
              <XORBenchmarkLab lang={lang} />
            )}

            {/* Sub-view 2: Math Inspector Logger */}
            {rdSubTab === 'logger' && (
              <div className="space-y-6">
                <TrainingControls
                  lang={lang}
                  isTraining={isTraining}
                  onToggleTraining={() => setIsTraining(!isTraining)}
                  onStep={handleStep}
                  onReset={handleReset}
                  epochs={epochs}
                  currentLoss={currentLoss}
                  metrics={metrics}
                  learningRate={model.optimizer?.learning_rate || 0.01}
                  speed={speed}
                  setSpeed={setSpeed}
                  isConverged={metrics?.accuracy !== undefined && metrics.accuracy >= 0.99}
                />
                <MathInspectorLogger
                  logger={inspectorLogger}
                  model={model}
                  lang={lang}
                  onClearLogs={handleClearLogs}
                />
              </div>
            )}

            {/* Sub-view 3: Custom Activation Sandbox */}
            {rdSubTab === 'activation' && (
              <div className="space-y-6">
                <TrainingControls
                  lang={lang}
                  isTraining={isTraining}
                  onToggleTraining={() => setIsTraining(!isTraining)}
                  onStep={handleStep}
                  onReset={handleReset}
                  epochs={epochs}
                  currentLoss={currentLoss}
                  metrics={metrics}
                  learningRate={model.optimizer?.learning_rate || 0.01}
                  speed={speed}
                  setSpeed={setSpeed}
                  isConverged={metrics?.accuracy !== undefined && metrics.accuracy >= 0.99}
                />
                <CustomActivationLab
                  model={model}
                  lang={lang}
                  onActivationInjected={handleActivationInjected}
                />
              </div>
            )}
          </div>
        )}
      </main>

      {/* Neuron Inspector Modal */}
      {selectedNeuron && (
        <NeuronInspectorModal
          model={model}
          selectedNeuron={selectedNeuron}
          onClose={() => setSelectedNeuron(null)}
          lang={lang}
        />
      )}

      {/* Code Export Modal */}
      {showCodeModal && (
        <CodeExportModal
          model={model}
          dataset={dataset}
          lang={lang}
          onClose={() => setShowCodeModal(false)}
          onImportModel={handleImportModel}
        />
      )}

      {/* Math Guide Modal */}
      {showGuideModal && (
        <MathGuideModal
          lang={lang}
          onClose={() => setShowGuideModal(false)}
        />
      )}

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 px-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <span>MiniDeepEngine © 2026 • From Scratch Deep Learning Architecture</span>
          <span className="font-mono text-slate-400">
            z = W • a + b | a = σ(z) | Adam Optimizer
          </span>
        </div>
      </footer>
    </div>
  );
}
