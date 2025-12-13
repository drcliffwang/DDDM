import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Play, AlertCircle, TrendingUp } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface HoltWintersResult {
  column: string;
  total_observations: number;
  forecast_periods: number;
  seasonal_periods: number;
  trend_type: string;
  seasonal_type: string;
  metrics: {
    mae: number;
    mse: number;
    rmse: number;
    mape: number | null;
    aic: number | null;
    bic: number | null;
  };
  parameters: {
    alpha: number;
    beta: number | null;
    gamma: number | null;
  };
  historical: Array<{index: number; actual: number; fitted: number}>;
  forecast: Array<{index: number; forecast: number}>;
}

const HoltWinters: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [valueColumn, setValueColumn] = useState<string>('');
  const [seasonalPeriods, setSeasonalPeriods] = useState<number>(12);
  const [forecastPeriods, setForecastPeriods] = useState<number>(6);
  const [trend, setTrend] = useState<'add' | 'mul' | 'none'>('add');
  const [seasonal, setSeasonal] = useState<'add' | 'mul' | 'none'>('add');
  const [result, setResult] = useState<HoltWintersResult | null>(null);
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
      const response = await fetch(`${apiUrl}/holt-winters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          value_column: valueColumn,
          seasonal_periods: seasonalPeriods,
          forecast_periods: forecastPeriods,
          trend: trend === 'none' ? null : trend,
          seasonal: seasonal === 'none' ? null : seasonal
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Forecast failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run Holt-Winters forecast');
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
          <TrendingUp size={20} className="text-amber-500" />
          Holt-Winters 指數平滑設定
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">數值欄位</label>
            <select
              value={valueColumn}
              onChange={(e) => setValueColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500"
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
              value={seasonalPeriods}
              onChange={(e) => setSeasonalPeriods(parseInt(e.target.value) || 12)}
              min={2}
              max={365}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500"
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
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500"
            />
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">趨勢類型</label>
            <div className="flex gap-2">
              {(['add', 'mul', 'none'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTrend(t)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    trend === t ? 'bg-amber-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {t === 'add' ? '加法' : t === 'mul' ? '乘法' : '無'}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">季節類型</label>
            <div className="flex gap-2">
              {(['add', 'mul', 'none'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setSeasonal(s)}
                  className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    seasonal === s ? 'bg-amber-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {s === 'add' ? '加法' : s === 'mul' ? '乘法' : '無'}
                </button>
              ))}
            </div>
          </div>
        </div>
        
        <button
          onClick={runForecast}
          disabled={loading || !valueColumn}
          className="px-6 py-3 bg-amber-600 text-white font-medium rounded-lg hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '預測中...' : '執行預測'}
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
          {/* Metrics */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📊 模型評估指標</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">MAE</div>
                <div className="text-xl font-bold text-slate-800">{result.metrics.mae.toFixed(4)}</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">RMSE</div>
                <div className="text-xl font-bold text-slate-800">{result.metrics.rmse.toFixed(4)}</div>
              </div>
              {result.metrics.mape && (
                <div className="bg-amber-50 rounded-lg p-4 text-center">
                  <div className="text-xs text-slate-500">MAPE</div>
                  <div className="text-xl font-bold text-amber-700">{result.metrics.mape.toFixed(2)}%</div>
                </div>
              )}
              {result.metrics.aic && (
                <div className="bg-slate-50 rounded-lg p-4 text-center">
                  <div className="text-xs text-slate-500">AIC</div>
                  <div className="text-xl font-bold text-slate-800">{result.metrics.aic.toFixed(2)}</div>
                </div>
              )}
            </div>
            
            {/* Model Parameters */}
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="bg-blue-50 rounded p-3">
                <div className="text-slate-500">α (Level)</div>
                <div className="font-mono font-bold text-blue-700">{result.parameters.alpha.toFixed(4)}</div>
              </div>
              {result.parameters.beta !== null && (
                <div className="bg-green-50 rounded p-3">
                  <div className="text-slate-500">β (Trend)</div>
                  <div className="font-mono font-bold text-green-700">{result.parameters.beta.toFixed(4)}</div>
                </div>
              )}
              {result.parameters.gamma !== null && (
                <div className="bg-purple-50 rounded p-3">
                  <div className="text-slate-500">γ (Seasonal)</div>
                  <div className="font-mono font-bold text-purple-700">{result.parameters.gamma.toFixed(4)}</div>
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
                  <Line type="monotone" dataKey="actual" stroke="#3b82f6" name="實際值" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="fitted" stroke="#ef4444" name="擬合值" dot={false} strokeWidth={1} strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="forecast" stroke="#f59e0b" name="預測值" dot strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Forecast Table */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📋 預測數值</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm border border-slate-200">
                <thead className="bg-amber-50">
                  <tr>
                    <th className="px-4 py-2 border-b text-left">期數</th>
                    <th className="px-4 py-2 border-b text-right">預測值</th>
                  </tr>
                </thead>
                <tbody>
                  {result.forecast.map((f, idx) => (
                    <tr key={idx} className="hover:bg-amber-50/50">
                      <td className="px-4 py-2 border-b">T + {idx + 1}</td>
                      <td className="px-4 py-2 border-b text-right font-mono font-bold text-amber-700">
                        {f.forecast.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default HoltWinters;
