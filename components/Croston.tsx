import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { Play, AlertCircle, Zap } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface CrostonResult {
  column: string;
  alpha: number;
  total_observations: number;
  forecast_periods: number;
  demand_stats: {
    zero_periods: number;
    nonzero_periods: number;
    intermittence_ratio: number;
    avg_demand_size: number;
    avg_interval: number;
  };
  smoothed_values: {
    final_z: number;
    final_p: number;
    demand_rate: number;
  };
  metrics: {
    mae: number;
    rmse: number;
    mape: number | null;
  };
  historical: Array<{index: number; actual: number; fitted: number}>;
  forecast: Array<{index: number; forecast: number}>;
}

const Croston: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [valueColumn, setValueColumn] = useState<string>('');
  const [alpha, setAlpha] = useState<number>(0.1);
  const [forecastPeriods, setForecastPeriods] = useState<number>(6);
  const [result, setResult] = useState<CrostonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runForecast = async () => {
    if (!valueColumn) {
      setError('Please select a value column.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/croston`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          value_column: valueColumn,
          alpha,
          forecast_periods: forecastPeriods
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Forecast failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run Croston forecast');
    } finally {
      setLoading(false);
    }
  };

  // Combine historical and forecast data for chart
  const getChartData = () => {
    if (!result) return [];
    
    const data: Array<{index: number; actual?: number; fitted?: number; forecast?: number}> = [];
    
    // Add historical data
    for (const h of result.historical) {
      data.push({
        index: h.index,
        actual: h.actual,
        fitted: h.fitted
      });
    }
    
    // Add forecast data
    for (const f of result.forecast) {
      data.push({
        index: f.index,
        forecast: f.forecast
      });
    }
    
    return data;
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Zap size={20} className="text-orange-500" />
          Croston 間歇需求預測設定
        </h3>
        
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-4 text-sm text-orange-800">
          <strong>適用場景：</strong>間歇性需求資料（如備品、低頻銷售品），資料中包含大量零值。
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">需求欄位</label>
            <select
              value={valueColumn}
              onChange={(e) => setValueColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">平滑參數 α</label>
            <input
              type="number"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value) || 0.1)}
              min={0.01}
              max={1}
              step={0.05}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">預測期數</label>
            <input
              type="number"
              value={forecastPeriods}
              onChange={(e) => setForecastPeriods(parseInt(e.target.value) || 6)}
              min={1}
              max={60}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500"
            />
          </div>
        </div>
        
        <button
          onClick={runForecast}
          disabled={loading || !valueColumn}
          className="px-6 py-3 bg-orange-600 text-white font-medium rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '預測中...' : '執行 Croston 預測'}
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
          {/* Demand Statistics */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📊 需求特性分析</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">零需求期數</div>
                <div className="text-xl font-bold text-slate-800">{result.demand_stats.zero_periods}</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">有需求期數</div>
                <div className="text-xl font-bold text-slate-800">{result.demand_stats.nonzero_periods}</div>
              </div>
              <div className="bg-orange-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">間歇率</div>
                <div className="text-xl font-bold text-orange-700">{result.demand_stats.intermittence_ratio}%</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">平均需求量 (z)</div>
                <div className="text-xl font-bold text-slate-800">{result.demand_stats.avg_demand_size.toFixed(2)}</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">平均間隔 (p)</div>
                <div className="text-xl font-bold text-slate-800">{result.demand_stats.avg_interval.toFixed(2)}</div>
              </div>
            </div>
            
            {/* Croston Values */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">平滑需求量 (ẑ)</div>
                <div className="text-xl font-bold text-blue-700">{result.smoothed_values.final_z.toFixed(4)}</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">平滑間隔 (p̂)</div>
                <div className="text-xl font-bold text-green-700">{result.smoothed_values.final_p.toFixed(4)}</div>
              </div>
              <div className="bg-orange-100 rounded-lg p-4 text-center border-2 border-orange-300">
                <div className="text-xs text-slate-500">需求率 (ẑ/p̂)</div>
                <div className="text-2xl font-bold text-orange-700">{result.smoothed_values.demand_rate.toFixed(4)}</div>
              </div>
            </div>
          </div>
          
          {/* Metrics */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📈 模型評估</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">MAE</div>
                <div className="text-xl font-bold text-slate-800">{result.metrics.mae.toFixed(4)}</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">RMSE</div>
                <div className="text-xl font-bold text-slate-800">{result.metrics.rmse.toFixed(4)}</div>
              </div>
              {result.metrics.mape && (
                <div className="bg-orange-50 rounded-lg p-4 text-center">
                  <div className="text-xs text-slate-500">MAPE</div>
                  <div className="text-xl font-bold text-orange-700">{result.metrics.mape.toFixed(2)}%</div>
                </div>
              )}
            </div>
          </div>
          
          {/* Forecast Chart */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📈 預測圖表</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={getChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: '時間索引', position: 'insideBottomRight', offset: -5 }} />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <Legend />
                  <ReferenceLine y={result.smoothed_values.demand_rate} stroke="#f97316" strokeDasharray="5 5" label="需求率" />
                  <Line type="stepAfter" dataKey="actual" stroke="#3b82f6" name="實際需求" dot strokeWidth={2} />
                  <Line type="monotone" dataKey="fitted" stroke="#ef4444" name="擬合值" dot={false} strokeWidth={1} />
                  <Line type="monotone" dataKey="forecast" stroke="#f97316" name="預測值" dot strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Forecast Table */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📋 預測數值</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm border border-slate-200">
                <thead className="bg-orange-50">
                  <tr>
                    <th className="px-4 py-2 border-b text-left">期數</th>
                    <th className="px-4 py-2 border-b text-right">預測需求率</th>
                  </tr>
                </thead>
                <tbody>
                  {result.forecast.map((f, idx) => (
                    <tr key={idx} className="hover:bg-orange-50/50">
                      <td className="px-4 py-2 border-b">T + {idx + 1}</td>
                      <td className="px-4 py-2 border-b text-right font-mono font-bold text-orange-700">
                        {f.forecast.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-sm text-slate-500">
                Croston 預測為常數需求率，適用於間歇性需求的庫存規劃。
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Croston;
