import React from 'react';

export type TimeSeriesType = 
  | 'holt-winters' 
  | 'decomposition' 
  | 'arima' 
  | 'croston';

interface Props {
  activeModel: TimeSeriesType;
  onModelChange: (model: TimeSeriesType) => void;
}

const TimeSeriesMenu: React.FC<Props> = ({ activeModel, onModelChange }) => {
  const models: { id: TimeSeriesType; label: string }[] = [
    { id: 'holt-winters', label: 'Holt-Winters' },
    { id: 'decomposition', label: '趨勢 + 季節分解' },
    { id: 'arima', label: 'ARIMA' },
    { id: 'croston', label: 'Croston 間歇需求' },
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
                    ? 'bg-amber-500 text-white shadow-md shadow-amber-200' 
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

export default TimeSeriesMenu;
