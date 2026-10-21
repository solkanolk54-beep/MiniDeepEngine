import React from 'react';
import { Cpu, BookOpen, Download, Code2, Globe, Sparkles, RefreshCw } from 'lucide-react';
import { Language, translations } from '../translations';
import { Dataset, DATASET_PRESETS } from '../engine/datasets';

interface HeaderProps {
  lang: Language;
  setLang: (lang: Language) => void;
  activeDataset: Dataset;
  onSelectDataset: (d: Dataset) => void;
  onOpenGuide: () => void;
  onOpenCode: () => void;
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  lang,
  setLang,
  activeDataset,
  onSelectDataset,
  onOpenGuide,
  onOpenCode,
  onReset,
}) => {
  const t = translations[lang];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30 px-4 py-3">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-cyan-400 p-[2px] shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Cpu className="w-6 h-6 text-indigo-400 animate-pulse" />
            </div>
            <span className="absolute -bottom-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                MiniDeepEngine
                <span className="text-xs px-2 py-0.5 rounded-full font-mono font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  v1.0 From Scratch
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400 line-clamp-1 max-w-md">
              {t.appSubtitle}
            </p>
          </div>
        </div>

        {/* Dataset Quick Picker & Action Buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700/60 rounded-lg px-2.5 py-1.5 text-xs text-slate-300">
            <span className="text-slate-400 font-medium hidden sm:inline">{t.selectDataset}:</span>
            <select
              value={activeDataset.id}
              onChange={(e) => {
                const found = DATASET_PRESETS.find(d => d.id === e.target.value);
                if (found) onSelectDataset(found);
              }}
              className="bg-transparent font-medium text-white focus:outline-none cursor-pointer"
            >
              {DATASET_PRESETS.map((d) => (
                <option key={d.id} value={d.id} className="bg-slate-900 text-white">
                  {lang === 'ar' ? d.nameAr : d.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={onReset}
            title={t.reset}
            className="p-2 text-slate-400 hover:text-white bg-slate-800/80 hover:bg-slate-700 border border-slate-700/60 rounded-lg transition"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          <button
            onClick={onOpenCode}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/60 border border-indigo-500/30 rounded-lg transition shadow-sm"
          >
            <Code2 className="w-4 h-4" />
            <span>Python / JSON</span>
          </button>

          <button
            onClick={onOpenGuide}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-emerald-300 bg-emerald-950/60 hover:bg-emerald-900/60 border border-emerald-500/30 rounded-lg transition shadow-sm"
          >
            <BookOpen className="w-4 h-4" />
            <span>{lang === 'ar' ? 'الدليل الرياضي' : 'Math Guide'}</span>
          </button>

          {/* Language Switch */}
          <button
            onClick={() => setLang(lang === 'ar' ? 'en' : 'ar')}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 border border-slate-700 rounded-lg transition"
            title="تبديل اللغة / Toggle Language"
          >
            <Globe className="w-4 h-4 text-cyan-400" />
            <span>{lang === 'ar' ? 'English' : 'العربية'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
