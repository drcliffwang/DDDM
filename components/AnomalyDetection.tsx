import React, { useState } from 'react';
import { Play, AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface AnomalyResult {
  column: string;
  method: string;
  threshold: number;
  total_rows: number;
  anomaly_count: number;
  anomaly_percentage: number;
  statistics: {
    mean: number;
    std: number;
    q1: number;
    q3: number;
    iqr: number;
    lower_bound: number;
    upper_bound: number;
  };
  anomalies: Array<{
    row_index: number;
    value: number;
    row_data: Record<string, string>;
  }>;
}

const AnomalyDetection: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [column, setColumn] = useState<string>('');
  const [method, setMethod] = useState<'zscore' | 'iqr'>('zscore');
  const [threshold, setThreshold] = useState<number>(3);
  const [result, setResult] = useState<AnomalyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runDetection = async () => {
    if (!column) {
      setError('Please select a column.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/anomaly-detection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          column,
          method,
          threshold
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Detection failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run anomaly detection');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <AlertTriangle size={20} className="text-amber-500" />
          異常偵測設定
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">選擇欄位</label>
            <select
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500"
            >
              <option value="">選擇數值欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">偵測方法</label>
            <div className="flex gap-2">
              <button
                onClick={() => setMethod('zscore')}
                className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  method === 'zscore' ? 'bg-amber-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                Z-Score
              </button>
              <button
                onClick={() => setMethod('iqr')}
                className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  method === 'iqr' ? 'bg-amber-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                IQR
              </button>
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              臨界值 ({method === 'zscore' ? 'σ' : 'IQR 倍數'})
            </label>
            <input
              type="number"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value) || 3)}
              step="0.5"
              min="1"
              max="5"
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500"
            />
          </div>
        </div>
        
        <p className="text-sm text-slate-500 mb-4">
          {method === 'zscore' 
            ? `Z-Score 方法：偵測超過平均值 ±${threshold} 個標準差的數據點`
            : `IQR 方法：偵測超過 Q1 - ${threshold}×IQR 或 Q3 + ${threshold}×IQR 的數據點`
          }
        </p>
        
        <button
          onClick={runDetection}
          disabled={loading || !column}
          className="px-6 py-3 bg-amber-600 text-white font-medium rounded-lg hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '偵測中...' : '執行異常偵測'}
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
          {/* Summary */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📊 偵測結果摘要</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">總資料點</div>
                <div className="text-xl font-bold text-slate-800">{result.total_rows}</div>
              </div>
              <div className={`rounded-lg p-4 text-center ${result.anomaly_count > 0 ? 'bg-red-50' : 'bg-green-50'}`}>
                <div className="text-xs text-slate-500">異常點數</div>
                <div className={`text-xl font-bold ${result.anomaly_count > 0 ? 'text-red-700' : 'text-green-700'}`}>
                  {result.anomaly_count}
                </div>
              </div>
              <div className={`rounded-lg p-4 text-center ${result.anomaly_percentage > 5 ? 'bg-amber-50' : 'bg-slate-50'}`}>
                <div className="text-xs text-slate-500">異常比例</div>
                <div className={`text-xl font-bold ${result.anomaly_percentage > 5 ? 'text-amber-700' : 'text-slate-800'}`}>
                  {result.anomaly_percentage}%
                </div>
              </div>
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">正常範圍</div>
                <div className="text-sm font-bold text-blue-700">
                  {result.statistics.lower_bound.toFixed(2)} ~ {result.statistics.upper_bound.toFixed(2)}
                </div>
              </div>
            </div>
            
            {/* Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
              <div className="bg-slate-50 rounded p-3">
                <div className="text-slate-500">平均值</div>
                <div className="font-mono">{result.statistics.mean.toFixed(4)}</div>
              </div>
              <div className="bg-slate-50 rounded p-3">
                <div className="text-slate-500">標準差</div>
                <div className="font-mono">{result.statistics.std.toFixed(4)}</div>
              </div>
              <div className="bg-slate-50 rounded p-3">
                <div className="text-slate-500">Q1 (25%)</div>
                <div className="font-mono">{result.statistics.q1.toFixed(4)}</div>
              </div>
              <div className="bg-slate-50 rounded p-3">
                <div className="text-slate-500">Q3 (75%)</div>
                <div className="font-mono">{result.statistics.q3.toFixed(4)}</div>
              </div>
              <div className="bg-slate-50 rounded p-3">
                <div className="text-slate-500">IQR</div>
                <div className="font-mono">{result.statistics.iqr.toFixed(4)}</div>
              </div>
            </div>
          </div>
          
          {/* Anomaly List */}
          {result.anomaly_count > 0 ? (
            <div className="bg-white rounded-xl p-6 border border-slate-200">
              <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                <AlertTriangle className="text-red-500" size={20} />
                偵測到的異常資料 ({result.anomaly_count})
              </h3>
              
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm border border-slate-200">
                  <thead className="bg-red-50">
                    <tr>
                      <th className="px-4 py-2 border-b text-left">列索引</th>
                      <th className="px-4 py-2 border-b text-left">{result.column} (異常值)</th>
                      {Object.keys(result.anomalies[0]?.row_data || {}).slice(0, 5).map((key) => (
                        <th key={key} className="px-4 py-2 border-b text-left">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.anomalies.slice(0, 20).map((a, idx) => (
                      <tr key={idx} className="hover:bg-red-50/50">
                        <td className="px-4 py-2 border-b">{a.row_index + 1}</td>
                        <td className="px-4 py-2 border-b font-bold text-red-700">{a.value.toFixed(4)}</td>
                        {Object.entries(a.row_data).slice(0, 5).map(([, val], i) => (
                          <td key={i} className="px-4 py-2 border-b">{val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.anomaly_count > 20 && (
                  <p className="mt-2 text-sm text-slate-500">僅顯示前 20 筆異常資料...</p>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-green-50 rounded-xl p-6 border border-green-200 flex items-center gap-4">
              <CheckCircle className="text-green-600" size={32} />
              <div>
                <h3 className="font-bold text-green-800">未偵測到異常資料</h3>
                <p className="text-green-700 text-sm">所有數據點都在正常範圍內。</p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AnomalyDetection;
