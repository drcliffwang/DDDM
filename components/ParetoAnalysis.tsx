import React, { useState } from 'react';
import { Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart, ReferenceLine } from 'recharts';
import { Play, AlertCircle } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface ParetoResult {
  pareto_data: Array<{
    category: string;
    value: number;
    percentage: number;
    cumulative_percentage: number;
  }>;
  total: number;
  threshold_80_index: number;
  categories_for_80: number;
  total_categories: number;
}

const ParetoAnalysis: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [categoryColumn, setCategoryColumn] = useState<string>('');
  const [valueColumn, setValueColumn] = useState<string>('');
  const [result, setResult] = useState<ParetoResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runAnalysis = async () => {
    if (!categoryColumn || !valueColumn) {
      setError('Please select both category and value columns.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/pareto-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          category_column: categoryColumn,
          value_column: valueColumn
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run Pareto Analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Column Selection */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4">📊 柏拉圖分析設定</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">類別欄位 (Category)</label>
            <select
              value={categoryColumn}
              onChange={(e) => setCategoryColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">數值欄位 (Value)</label>
            <select
              value={valueColumn}
              onChange={(e) => setValueColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          onClick={runAnalysis}
          disabled={loading || !categoryColumn || !valueColumn}
          className="mt-4 px-6 py-3 bg-emerald-600 text-white font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '分析中...' : '執行柏拉圖分析'}
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
          <h3 className="font-bold text-slate-900 mb-2">📈 柏拉圖圖表</h3>
          <p className="text-sm text-slate-500 mb-4">
            80% 的結果來自前 <span className="font-bold text-emerald-600">{result.categories_for_80}</span> 個類別 
            (共 {result.total_categories} 個)
          </p>
          
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={result.pareto_data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-45} textAnchor="end" height={80} />
                <YAxis yAxisId="left" label={{ value: '數值', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 100]} label={{ value: '累積 %', angle: 90, position: 'insideRight' }} />
                <Tooltip 
                  contentStyle={{ borderRadius: '8px' }}
                  formatter={(value: number, name: string) => [
                    name === 'value' ? value.toLocaleString() : `${value}%`,
                    name === 'value' ? '數值' : '累積百分比'
                  ]}
                />
                <Bar yAxisId="left" dataKey="value" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Line yAxisId="right" type="monotone" dataKey="cumulative_percentage" stroke="#f59e0b" strokeWidth={2} dot={{ fill: '#f59e0b', r: 4 }} />
                <ReferenceLine yAxisId="right" y={80} stroke="#ef4444" strokeDasharray="5 5" label={{ value: '80%', position: 'right', fill: '#ef4444' }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Summary */}
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="bg-emerald-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">總計</div>
              <div className="text-xl font-bold text-emerald-700">{result.total.toLocaleString()}</div>
            </div>
            <div className="bg-amber-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">80% 門檻類別數</div>
              <div className="text-xl font-bold text-amber-700">{result.categories_for_80}</div>
            </div>
            <div className="bg-slate-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">總類別數</div>
              <div className="text-xl font-bold text-slate-700">{result.total_categories}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ParetoAnalysis;
