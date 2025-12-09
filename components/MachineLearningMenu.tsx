import React from 'react';

export type MLModel = 
  | 'logistic-regression' 
  | 'decision-tree' 
  | 'pca' 
  | 'factor-analysis' 
  | 'kmeans' 
  | 'hierarchical' 
  | 'apriori' 
  | 'random-forest' 
  | 'neural-network';

interface Props {
  activeModel: MLModel;
  onModelChange: (model: MLModel) => void;
}

const MachineLearningMenu: React.FC<Props> = ({ activeModel, onModelChange }) => {
  const models: { id: MLModel; label: string }[] = [
    { id: 'logistic-regression', label: '邏輯斯迴歸' },
    { id: 'decision-tree', label: '決策樹' },
    { id: 'pca', label: '主成分分析 PCA' },
    { id: 'factor-analysis', label: '因素分析 FA' },
    { id: 'kmeans', label: 'K-Means 集群' },
    { id: 'hierarchical', label: '階層式集群' },
    { id: 'apriori', label: '關聯規則 Apriori' },
    { id: 'random-forest', label: '隨機森林 RF' },
    { id: 'neural-network', label: '神經網絡' },
  ];

  return (
    <div className="bg-white border-b border-slate-200 py-3 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex gap-3 overflow-x-auto no-scrollbar pb-1">
          {models.map((model) => {
            const isActive = activeModel === model.id;
            return (
              <button
                key={model.id}
                onClick={() => onModelChange(model.id)}
                className={`
                  px-4 py-2 rounded-full text-sm font-medium transition-all whitespace-nowrap
                  ${isActive 
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-200' 
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }
                `}
              >
                {model.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default MachineLearningMenu;
