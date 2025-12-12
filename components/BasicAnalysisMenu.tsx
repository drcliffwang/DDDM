import React from 'react';

export type BasicAnalysisType = 
  | 'anomaly-detection' 
  | 'pareto-analysis' 
  | 'hypothesis-testing' 
  | 'chi-square' 
  | 'regression' 
  | 'correlation-matrix' 
  | 'word-cloud';

interface Props {
  activeAnalysis: BasicAnalysisType;
  onAnalysisChange: (analysis: BasicAnalysisType) => void;
}

const BasicAnalysisMenu: React.FC<Props> = ({ activeAnalysis, onAnalysisChange }) => {
  const analyses: { id: BasicAnalysisType; label: string }[] = [
    { id: 'anomaly-detection', label: '異常偵測' },
    { id: 'pareto-analysis', label: '柏拉圖分析' },
    { id: 'hypothesis-testing', label: '假設檢定' },
    { id: 'chi-square', label: '卡方檢定 X²' },
    { id: 'regression', label: '迴歸分析' },
    { id: 'correlation-matrix', label: '相關係數矩陣' },
    { id: 'word-cloud', label: '文字雲' },
  ];

  return (
    <div className="bg-white border-b border-slate-200 py-3 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex gap-3 overflow-x-auto no-scrollbar pb-1">
          {analyses.map((analysis) => {
            const isActive = activeAnalysis === analysis.id;
            return (
              <button
                key={analysis.id}
                onClick={() => onAnalysisChange(analysis.id)}
                className={`
                  px-4 py-2 rounded-full text-sm font-medium transition-all whitespace-nowrap
                  ${isActive 
                    ? 'bg-emerald-600 text-white shadow-md shadow-emerald-200' 
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }
                `}
              >
                {analysis.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default BasicAnalysisMenu;
