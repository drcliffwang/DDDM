import React, { useState, useEffect } from 'react';
import { Settings, Play, Check } from 'lucide-react';

interface Props {
  headers: string[];
  onRunAnalysis: (target: string, features: string[]) => void;
  isProcessing: boolean;
}

const FeatureSelection: React.FC<Props> = ({ headers, onRunAnalysis, isProcessing }) => {
  const [target, setTarget] = useState<string>('');
  const [features, setFeatures] = useState<string[]>([]);

  // Reset selections when dataset changes (headers change)
  useEffect(() => {
    setTarget('');
    setFeatures([]);
  }, [headers]);

  const handleFeatureToggle = (header: string) => {
    if (features.includes(header)) {
      setFeatures(features.filter(f => f !== header));
    } else {
      setFeatures([...features, header]);
    }
  };

  const isValid = target && features.length > 0 && !features.includes(target);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-orange-500">
      <div className="p-8">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-sm font-bold text-slate-600">2</div>
              Model Configuration
            </h2>
            <p className="mt-1 text-slate-500 ml-10">Select your dependent variable (Y) and independent variables (Xs)</p>
          </div>
        </div>

        <div className="mt-8 ml-10 grid grid-cols-1 lg:grid-cols-2 gap-12">
          
          {/* Target Selection */}
          <div className="space-y-4">
            <h3 className="font-semibold text-slate-900 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-orange-500"></span>
              Target Variable (Y)
            </h3>
            <div className="space-y-2 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
              {headers.map(header => (
                <button
                  key={`target-${header}`}
                  onClick={() => {
                    setTarget(header);
                    if (features.includes(header)) setFeatures(features.filter(f => f !== header));
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg border transition-all ${
                    target === header 
                      ? 'border-orange-500 bg-orange-50 text-orange-700 font-medium shadow-sm' 
                      : 'border-slate-200 hover:border-slate-300 text-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span>{header}</span>
                    {target === header && <Check size={16} />}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Feature Selection */}
          <div className="space-y-4">
            <h3 className="font-semibold text-slate-900 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
              Features (Xs)
            </h3>
            <div className="space-y-2 max-h-60 overflow-y-auto pr-2 custom-scrollbar">
              {headers.map(header => {
                const isSelected = features.includes(header);
                const isTarget = target === header;
                return (
                  <button
                    key={`feat-${header}`}
                    disabled={isTarget}
                    onClick={() => handleFeatureToggle(header)}
                    className={`w-full text-left px-4 py-3 rounded-lg border transition-all ${
                      isTarget 
                        ? 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed'
                        : isSelected
                          ? 'border-indigo-500 bg-indigo-50 text-indigo-700 font-medium shadow-sm' 
                          : 'border-slate-200 hover:border-slate-300 text-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{header}</span>
                      {isSelected && <Check size={16} />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="mt-8 ml-10 pt-6 border-t border-slate-100 flex justify-end">
          <button
            onClick={() => onRunAnalysis(target, features)}
            disabled={!isValid || isProcessing}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-bold text-white shadow-lg transition-all transform active:scale-95 ${
              isValid && !isProcessing
                ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 hover:shadow-xl hover:from-indigo-500 hover:to-indigo-600' 
                : 'bg-slate-300 cursor-not-allowed shadow-none'
            }`}
          >
            <Play size={18} fill="currentColor" />
            {isProcessing ? 'Training Model...' : 'Train Random Forest'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FeatureSelection;