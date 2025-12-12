import React from 'react';
import { Search, BarChart2, Clock, BrainCircuit } from 'lucide-react';

export type AnalysisTab = 'data-exploration' | 'basic-analysis' | 'time-series' | 'machine-learning';

interface Props {
  activeTab: AnalysisTab;
  onTabChange: (tab: AnalysisTab) => void;
}

const AnalysisTabs: React.FC<Props> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'data-exploration', label: '資料探索', icon: <Search size={26} /> },
    { id: 'basic-analysis', label: '基本分析', icon: <BarChart2 size={26} /> },
    { id: 'time-series', label: '時間序列', icon: <Clock size={26} /> },
    { id: 'machine-learning', label: '機器學習', icon: <BrainCircuit size={26} /> },
  ];

  return (
    <div className="bg-white border-b border-slate-200 sticky top-16 z-40 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex space-x-4 overflow-x-auto no-scrollbar justify-center py-2">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            // Unique colors for each tab
            const colorMap: Record<string, { active: string; hover: string }> = {
              'data-exploration': { active: 'border-indigo-600 text-indigo-700 bg-indigo-50', hover: 'hover:text-indigo-700 hover:bg-indigo-50/50 hover:border-indigo-300' },
              'basic-analysis': { active: 'border-emerald-600 text-emerald-700 bg-emerald-50', hover: 'hover:text-emerald-700 hover:bg-emerald-50/50 hover:border-emerald-300' },
              'time-series': { active: 'border-amber-500 text-amber-700 bg-amber-50', hover: 'hover:text-amber-700 hover:bg-amber-50/50 hover:border-amber-300' },
              'machine-learning': { active: 'border-indigo-600 text-indigo-700 bg-indigo-50', hover: 'hover:text-indigo-700 hover:bg-indigo-50/50 hover:border-indigo-300' },
            };
            const colors = colorMap[tab.id] || colorMap['data-exploration'];
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id as AnalysisTab)}
                className={`
                  group flex items-center gap-3 py-5 px-8 border-b-4 text-xl font-bold transition-all duration-200 whitespace-nowrap rounded-t-xl
                  ${isActive 
                    ? colors.active 
                    : `border-transparent text-slate-500 ${colors.hover}`
                  }
                `}
              >
                {/* Icon with scale effect */}
                <span className={`transition-transform duration-300 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`}>
                  {tab.icon}
                </span>
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AnalysisTabs;
