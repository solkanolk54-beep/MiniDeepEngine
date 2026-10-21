import React, { useState } from 'react';
import { X, Copy, Download, Check, FileCode, Upload, Code } from 'lucide-react';
import { NeuralNetwork } from '../engine/model';
import { Dataset } from '../engine/datasets';
import { Language, translations } from '../translations';

interface CodeExportModalProps {
  model: NeuralNetwork;
  dataset: Dataset;
  lang: Language;
  onClose: () => void;
  onImportModel: (jsonStr: string) => void;
}

export const CodeExportModal: React.FC<CodeExportModalProps> = ({
  model,
  dataset,
  lang,
  onClose,
  onImportModel,
}) => {
  const t = translations[lang];
  const [tab, setTab] = useState<'python' | 'json'>('python');
  const [isCopied, setIsCopied] = useState(false);
  const [importJsonText, setImportJsonText] = useState('');
  const [importError, setImportError] = useState<string | null>(null);

  const pythonCode = model.exportPythonCode(dataset.X, dataset.y, 1000);
  const jsonModel = model.toJSON();

  const handleCopy = () => {
    const textToCopy = tab === 'python' ? pythonCode : jsonModel;
    navigator.clipboard.writeText(textToCopy);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleDownload = () => {
    const isPy = tab === 'python';
    const blob = new Blob([isPy ? pythonCode : jsonModel], {
      type: isPy ? 'text/x-python;charset=utf-8' : 'application/json;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = isPy ? 'minideepengine.py' : 'minideep_model.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleApplyImport = () => {
    try {
      setImportError(null);
      onImportModel(importJsonText);
      onClose();
    } catch (err: any) {
      setImportError(err.message || 'Invalid JSON model format');
    }
  };

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
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-3xl max-h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/10 border border-indigo-500/20 rounded-xl text-indigo-400">
              <FileCode className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                {lang === 'ar' ? 'تصدير الكود والنموذج البرمجي' : 'Code Generation & Model Export'}
              </h3>
              <p className="text-xs text-slate-400">
                {lang === 'ar'
                  ? 'كود بايثون متكامل ومستقل مبني بـ NumPy فقط أو ملف JSON لأوزان النموذج'
                  : 'Self-contained Python script using NumPy only, or JSON representation of trained weights'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setTab('python')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                tab === 'python'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              Python (.py)
            </button>
            <button
              onClick={() => setTab('json')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition ${
                tab === 'json'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              JSON Weights
            </button>
            <button
              onClick={onClose}
              aria-label="Close"
              className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition ml-2"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Action bar */}
        <div className="px-4 py-2 bg-slate-950 border-b border-slate-800 flex items-center justify-between gap-2 flex-wrap">
          <span className="text-xs text-slate-400 font-mono">
            {tab === 'python' ? 'minideepengine.py' : 'minideep_model.json'}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{isCopied ? t.copied : t.copyCode}</span>
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white shadow transition"
            >
              <Download className="w-3.5 h-3.5" />
              <span>{tab === 'python' ? t.downloadPython : t.exportJSON}</span>
            </button>
          </div>
        </div>

        {/* Code Content */}
        <div className="flex-1 overflow-y-auto p-4 bg-slate-950 font-mono text-xs" dir="ltr">
          {tab === 'python' ? (
            <pre className="text-slate-300 leading-relaxed whitespace-pre text-left selection:bg-indigo-500 selection:text-white font-mono">
              {pythonCode}
            </pre>
          ) : (
            <div className="space-y-4 text-left">
              <div>
                <span className="text-slate-400 text-xs font-sans block mb-1">
                  {lang === 'ar' ? 'الأوزان الحالية بتنسيق JSON:' : 'Current weights in JSON format:'}
                </span>
                <pre className="text-emerald-300 bg-slate-900 p-3 rounded-xl overflow-x-auto max-h-60 border border-slate-800 text-left font-mono">
                  {jsonModel}
                </pre>
              </div>

              {/* Import Section */}
              <div className="pt-2 border-t border-slate-800" dir={lang === 'ar' ? 'rtl' : 'ltr'}>
                <span className="text-slate-300 text-xs font-sans font-bold block mb-1">
                  {lang === 'ar' ? 'استيراد نموذج جديد (ألصق JSON هنا):' : 'Import Saved Model (Paste JSON below):'}
                </span>
                <textarea
                  value={importJsonText}
                  onChange={(e) => setImportJsonText(e.target.value)}
                  placeholder="Paste JSON model here..."
                  rows={4}
                  dir="ltr"
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl p-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono text-left"
                />
                {importError && (
                  <p className="text-xs text-rose-400 mt-1 font-sans">{importError}</p>
                )}
                <button
                  onClick={handleApplyImport}
                  disabled={!importJsonText.trim()}
                  className="mt-2 flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition"
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>{t.importJSON}</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
