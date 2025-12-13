import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar, ReferenceLine } from 'recharts';
import { Play, AlertCircle, Layers } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface STLResult {
  column: string;
  seasonal_period: number;
  total_observations: number;
  strength: {
    trend_strength: number;
    seasonal_strength: number;
  };
  statistics: {
    original_mean: number;
    original_std: number;
    trend_mean: number;
    seasonal_range: number;
    residual_std: number;
  };
  decomposition: Array<{index: number; original: number; trend: number; seasonal: number; residual: number}>;
  seasonal_pattern: Array<{period: number; value: number}>;
}

const STLDecomposition: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [valueColumn, setValueColumn] = useState<string>('');
  const [seasonalPeriod, setSeasonalPeriod] = useState<number>(12);
  const [result, setResult] = useState<STLResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runDecomposition = async () => {
    if (!valueColumn) {
      setError('Please select a value column.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/stl-decomposition`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          value_column: valueColumn,
          seasonal_period: seasonalPeriod
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Decomposition failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run STL decomposition');
    } finally {
      setLoading(false);
    }
  };

  const getStrengthColor = (strength: number) => {
    if (strength >= 0.7) return 'text-green-700 bg-green-50';
    if (strength >= 0.4) return 'text-amber-700 bg-amber-50';
    return 'text-slate-700 bg-slate-50';
  };

  const getStrengthLabel = (strength: number) => {
    if (strength >= 0.7) return '強';
    if (strength >= 0.4) return '中等';
    return '弱';
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Layers size={20} className="text-teal-500" />
          STL 分解設定 (趨勢 + 季節)
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">數值欄位</label>
            <select
              value={valueColumn}
              onChange={(e) => setValueColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-teal-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">季節週期</label>
            <input
              type="number"
              value={seasonalPeriod}
              onChange={(e) => setSeasonalPeriod(parseInt(e.target.value) || 12)}
              min={2}
              max={365}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-teal-500"
            />
            <p className="text-xs text-slate-500 mt-1">例：月資料=12，季資料=4，週資料=7</p>
          </div>
        </div>
        
        <button
          onClick={runDecomposition}
          disabled={loading || !valueColumn}
          className="px-6 py-3 bg-teal-600 text-white font-medium rounded-lg hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '分解中...' : '執行 STL 分解'}
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
        <>
          {/* Strength Metrics */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📊 分解強度</h3>
            
            <div className="grid grid-cols-2 gap-6">
              <div className={`rounded-lg p-4 text-center ${getStrengthColor(result.strength.trend_strength)}`}>
                <div className="text-xs text-slate-500">趨勢強度</div>
                <div className="text-3xl font-bold">{(result.strength.trend_strength * 100).toFixed(1)}%</div>
                <div className="text-sm mt-1">{getStrengthLabel(result.strength.trend_strength)}</div>
              </div>
              <div className={`rounded-lg p-4 text-center ${getStrengthColor(result.strength.seasonal_strength)}`}>
                <div className="text-xs text-slate-500">季節強度</div>
                <div className="text-3xl font-bold">{(result.strength.seasonal_strength * 100).toFixed(1)}%</div>
                <div className="text-sm mt-1">{getStrengthLabel(result.strength.seasonal_strength)}</div>
              </div>
            </div>
            
            {/* Statistics */}
            <div className="grid grid-cols-5 gap-3 mt-4 text-sm">
              <div className="bg-slate-50 rounded p-3 text-center">
                <div className="text-slate-500">原始平均</div>
                <div className="font-mono font-bold">{result.statistics.original_mean.toFixed(2)}</div>
              </div>
              <div className="bg-slate-50 rounded p-3 text-center">
                <div className="text-slate-500">原始標準差</div>
                <div className="font-mono font-bold">{result.statistics.original_std.toFixed(2)}</div>
              </div>
              <div className="bg-blue-50 rounded p-3 text-center">
                <div className="text-slate-500">趨勢平均</div>
                <div className="font-mono font-bold text-blue-700">{result.statistics.trend_mean.toFixed(2)}</div>
              </div>
              <div className="bg-green-50 rounded p-3 text-center">
                <div className="text-slate-500">季節範圍</div>
                <div className="font-mono font-bold text-green-700">{result.statistics.seasonal_range.toFixed(2)}</div>
              </div>
              <div className="bg-slate-50 rounded p-3 text-center">
                <div className="text-slate-500">殘差標準差</div>
                <div className="font-mono font-bold">{result.statistics.residual_std.toFixed(2)}</div>
              </div>
            </div>
          </div>
          
          {/* Original + Trend Chart */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📈 原始資料 vs 趨勢</h3>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.decomposition} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <Legend />
                  <Line type="monotone" dataKey="original" stroke="#64748b" name="原始資料" dot={false} strokeWidth={1} />
                  <Line type="monotone" dataKey="trend" stroke="#0d9488" name="趨勢" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Seasonal Chart */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">🔄 季節成分</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.decomposition} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <ReferenceLine y={0} stroke="#94a3b8" />
                  <Line type="monotone" dataKey="seasonal" stroke="#22c55e" name="季節成分" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Seasonal Pattern (One Cycle) */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📅 季節模式 (單週期)</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={result.seasonal_pattern} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" label={{ value: '週期位置', position: 'insideBottomRight', offset: -5 }} />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <ReferenceLine y={0} stroke="#94a3b8" />
                  <Bar dataKey="value" name="季節效應" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Residual Chart */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📉 殘差成分</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.decomposition} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <ReferenceLine y={0} stroke="#94a3b8" />
                  <Line type="monotone" dataKey="residual" stroke="#f59e0b" name="殘差" dot={false} strokeWidth={1} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default STLDecomposition;
