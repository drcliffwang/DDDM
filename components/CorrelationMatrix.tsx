import React, { useState } from 'react';
import { Play, AlertCircle } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface CorrelationResult {
  columns: string[];
  matrix_data: Array<{row: string; col: string; value: number}>;
  n_observations: number;
}

const CorrelationMatrix: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [result, setResult] = useState<CorrelationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const toggleColumn = (col: string) => {
    if (selectedColumns.includes(col)) {
      setSelectedColumns(selectedColumns.filter(c => c !== col));
    } else {
      setSelectedColumns([...selectedColumns, col]);
    }
  };

  const selectAll = () => {
    setSelectedColumns([...dataset.headers]);
  };

  const clearAll = () => {
    setSelectedColumns([]);
  };

  const runAnalysis = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/correlation-matrix`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          columns: selectedColumns.length > 0 ? selectedColumns : null
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to compute correlation matrix');
    } finally {
      setLoading(false);
    }
  };

  const getCorrelationColor = (value: number) => {
    if (value >= 0.7) return 'bg-green-500 text-white';
    if (value >= 0.4) return 'bg-green-300 text-green-900';
    if (value >= 0.2) return 'bg-green-100 text-green-800';
    if (value > -0.2) return 'bg-slate-100 text-slate-700';
    if (value > -0.4) return 'bg-red-100 text-red-800';
    if (value > -0.7) return 'bg-red-300 text-red-900';
    return 'bg-red-500 text-white';
  };

  const getCorrelationValue = (row: string, col: string): number | null => {
    if (!result) return null;
    const cell = result.matrix_data.find(d => d.row === row && d.col === col);
    return cell ? cell.value : null;
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4">📊 相關係數矩陣設定</h3>
        <p className="text-sm text-slate-500 mb-4">選擇要計算相關係數的欄位 (僅數值欄位有效)</p>
        
        <div className="flex gap-2 mb-3">
          <button onClick={selectAll} className="px-3 py-1 text-sm bg-slate-100 rounded hover:bg-slate-200">全選</button>
          <button onClick={clearAll} className="px-3 py-1 text-sm bg-slate-100 rounded hover:bg-slate-200">清除</button>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {dataset.headers.map((h) => (
            <button
              key={h}
              onClick={() => toggleColumn(h)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                selectedColumns.includes(h)
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {h}
            </button>
          ))}
        </div>
        
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="px-6 py-3 bg-emerald-600 text-white font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '計算中...' : '計算相關係數矩陣'}
        </button>
        
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
            <AlertCircle size={18} />
            {error}
          </div>
        )}
      </div>
      
      {/* Results */}
      {result && (
        <div className="bg-white rounded-xl p-6 border border-slate-200">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-slate-900">📈 相關係數矩陣</h3>
            <span className="text-sm text-slate-500">觀察數: {result.n_observations}</span>
          </div>
          
          {/* Heatmap Legend */}
          <div className="flex items-center gap-2 mb-4 text-xs">
            <span className="text-slate-500">負相關</span>
            <div className="flex">
              <span className="w-6 h-4 bg-red-500"></span>
              <span className="w-6 h-4 bg-red-300"></span>
              <span className="w-6 h-4 bg-red-100"></span>
              <span className="w-6 h-4 bg-slate-100"></span>
              <span className="w-6 h-4 bg-green-100"></span>
              <span className="w-6 h-4 bg-green-300"></span>
              <span className="w-6 h-4 bg-green-500"></span>
            </div>
            <span className="text-slate-500">正相關</span>
          </div>
          
          {/* Matrix Table */}
          <div className="overflow-x-auto">
            <table className="text-sm border-collapse">
              <thead>
                <tr>
                  <th className="px-3 py-2 bg-slate-50 border border-slate-200"></th>
                  {result.columns.map((col) => (
                    <th key={col} className="px-3 py-2 bg-slate-50 border border-slate-200 text-center font-medium" style={{writingMode: 'vertical-lr', maxWidth: '40px', minHeight: '60px'}}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.columns.map((row) => (
                  <tr key={row}>
                    <td className="px-3 py-2 bg-slate-50 border border-slate-200 font-medium whitespace-nowrap">{row}</td>
                    {result.columns.map((col) => {
                      const value = getCorrelationValue(row, col);
                      return (
                        <td
                          key={col}
                          className={`px-3 py-2 border border-slate-200 text-center font-mono text-xs ${value !== null ? getCorrelationColor(value) : ''}`}
                          title={`${row} ↔ ${col}: ${value?.toFixed(4)}`}
                        >
                          {value !== null ? value.toFixed(2) : '-'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Interpretation */}
          <div className="mt-4 p-4 bg-slate-50 rounded-lg text-sm text-slate-600">
            <strong>解讀指南:</strong>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>|r| ≥ 0.7: 強相關</li>
              <li>0.4 ≤ |r| &lt; 0.7: 中等相關</li>
              <li>0.2 ≤ |r| &lt; 0.4: 弱相關</li>
              <li>|r| &lt; 0.2: 可忽略</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default CorrelationMatrix;
