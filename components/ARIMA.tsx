import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, ComposedChart } from 'recharts';
import { Play, AlertCircle, Activity } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface ARIMAResult {
  column: string;
  order: {p: number; d: number; q: number};
  total_observations: number;
  forecast_periods: number;
  metrics: {
    mae: number;
    mse: number;
    rmse: number;
    mape: number | null;
    aic: number;
    bic: number;
  };
  historical: Array<{index: number; actual: number; fitted?: number}>;
  forecast: Array<{index: number; forecast: number; lower_ci: number; upper_ci: number}>;
}

const ARIMA: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [valueColumn, setValueColumn] = useState<string>('');
  const [pOrder, setPOrder] = useState<number>(1);
  const [dOrder, setDOrder] = useState<number>(1);
  const [qOrder, setQOrder] = useState<number>(1);
  const [forecastPeriods, setForecastPeriods] = useState<number>(6);
  const [result, setResult] = useState<ARIMAResult | null>(null);
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
      const response = await fetch(`${apiUrl}/arima`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          value_column: valueColumn,
          p: pOrder,
          d: dOrder,
          q: qOrder,
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
      setError(e.message || 'Failed to run ARIMA forecast');
    } finally {
      setLoading(false);
    }
  };

  // Combine historical and forecast data for chart
  const getChartData = () => {
    if (!result) return [];
    
    const data: Array<{index: number; actual?: number; fitted?: number; forecast?: number; lower_ci?: number; upper_ci?: number}> = [];
    
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
        forecast: f.forecast,
        lower_ci: f.lower_ci,
        upper_ci: f.upper_ci
      });
    }
    
    return data;
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Activity size={20} className="text-purple-500" />
          ARIMA 模型設定
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">數值欄位</label>
            <select
              value={valueColumn}
              onChange={(e) => setValueColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">預測期數</label>
            <input
              type="number"
              value={forecastPeriods}
              onChange={(e) => setForecastPeriods(parseInt(e.target.value) || 6)}
              min={1}
              max={60}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            />
          </div>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">ARIMA 參數 (p, d, q)</label>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-slate-500 mb-1">p (自迴歸)</label>
              <input
                type="number"
                value={pOrder}
                onChange={(e) => setPOrder(parseInt(e.target.value) || 0)}
                min={0}
                max={10}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-center font-mono text-lg"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-500 mb-1">d (差分)</label>
              <input
                type="number"
                value={dOrder}
                onChange={(e) => setDOrder(parseInt(e.target.value) || 0)}
                min={0}
                max={3}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-center font-mono text-lg"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-500 mb-1">q (移動平均)</label>
              <input
                type="number"
                value={qOrder}
                onChange={(e) => setQOrder(parseInt(e.target.value) || 0)}
                min={0}
                max={10}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-center font-mono text-lg"
              />
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            p: AR階數 (過去值的影響) | d: 差分次數 (平穩化) | q: MA階數 (過去誤差的影響)
          </p>
        </div>
        
        <button
          onClick={runForecast}
          disabled={loading || !valueColumn}
          className="px-6 py-3 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '預測中...' : '執行 ARIMA 預測'}
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
            <h3 className="font-bold text-slate-900 mb-4">
              📊 ARIMA({result.order.p}, {result.order.d}, {result.order.q}) 模型評估
            </h3>
            
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
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <div className="text-xs text-slate-500">MAPE</div>
                  <div className="text-xl font-bold text-purple-700">{result.metrics.mape.toFixed(2)}%</div>
                </div>
              )}
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">AIC / BIC</div>
                <div className="text-lg font-bold text-slate-800">{result.metrics.aic.toFixed(0)} / {result.metrics.bic.toFixed(0)}</div>
              </div>
            </div>
          </div>
          
          {/* Forecast Chart */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📈 預測圖表 (含 95% 信賴區間)</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={getChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: '時間索引', position: 'insideBottomRight', offset: -5 }} />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <Legend />
                  <Area type="monotone" dataKey="upper_ci" stroke="none" fill="#c4b5fd" name="95% CI 上界" />
                  <Area type="monotone" dataKey="lower_ci" stroke="none" fill="#c4b5fd" name="95% CI 下界" />
                  <Line type="monotone" dataKey="actual" stroke="#3b82f6" name="實際值" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="fitted" stroke="#ef4444" name="擬合值" dot={false} strokeWidth={1} strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="forecast" stroke="#8b5cf6" name="預測值" dot strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Forecast Table */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📋 預測數值</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm border border-slate-200">
                <thead className="bg-purple-50">
                  <tr>
                    <th className="px-4 py-2 border-b text-left">期數</th>
                    <th className="px-4 py-2 border-b text-right">預測值</th>
                    <th className="px-4 py-2 border-b text-right">95% CI 下界</th>
                    <th className="px-4 py-2 border-b text-right">95% CI 上界</th>
                  </tr>
                </thead>
                <tbody>
                  {result.forecast.map((f, idx) => (
                    <tr key={idx} className="hover:bg-purple-50/50">
                      <td className="px-4 py-2 border-b">T + {idx + 1}</td>
                      <td className="px-4 py-2 border-b text-right font-mono font-bold text-purple-700">
                        {f.forecast.toFixed(4)}
                      </td>
                      <td className="px-4 py-2 border-b text-right font-mono text-slate-500">
                        {f.lower_ci.toFixed(4)}
                      </td>
                      <td className="px-4 py-2 border-b text-right font-mono text-slate-500">
                        {f.upper_ci.toFixed(4)}
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

export default ARIMA;
