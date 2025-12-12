import React, { useState } from 'react';
import { Play, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface TestResult {
  test_type: string;
  test_type_zh: string;
  t_statistic: number;
  p_value: number;
  significant_005: boolean;
  significant_001: boolean;
  [key: string]: any;
}

const HypothesisTesting: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [testType, setTestType] = useState<'one-sample' | 'two-sample' | 'paired'>('one-sample');
  const [column1, setColumn1] = useState<string>('');
  const [column2, setColumn2] = useState<string>('');
  const [hypothesizedMean, setHypothesizedMean] = useState<number>(0);
  const [result, setResult] = useState<TestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runTest = async () => {
    if (!column1) {
      setError('Please select the first column.');
      return;
    }
    if ((testType === 'two-sample' || testType === 'paired') && !column2) {
      setError('Please select the second column.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/hypothesis-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          test_type: testType,
          column1,
          column2: testType !== 'one-sample' ? column2 : null,
          hypothesized_mean: hypothesizedMean
        })
      });
      
      if (!response.ok) {
        let errMsg = 'Test failed';
        try {
          const errData = await response.json();
          errMsg = errData.detail || errMsg;
        } catch {
          errMsg = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errMsg);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      const msg = e?.message || (typeof e === 'string' ? e : 'Failed to run hypothesis test');
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Test Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4">🧪 假設檢定設定</h3>
        
        {/* Test Type Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">檢定類型</label>
          <div className="flex gap-2">
            {[
              { id: 'one-sample', label: '單樣本 T 檢定' },
              { id: 'two-sample', label: '雙樣本 T 檢定' },
              { id: 'paired', label: '配對 T 檢定' }
            ].map((t) => (
              <button
                key={t.id}
                onClick={() => setTestType(t.id as any)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  testType === t.id
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              {testType === 'one-sample' ? '樣本欄位' : testType === 'two-sample' ? '數值欄位 (Y)' : '樣本 1 欄位'}
            </label>
            <select
              value={column1}
              onChange={(e) => setColumn1(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          
          {testType === 'one-sample' ? (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">假設平均值 (μ₀)</label>
              <input
                type="number"
                value={hypothesizedMean}
                onChange={(e) => setHypothesizedMean(parseFloat(e.target.value) || 0)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                placeholder="0"
              />
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                {testType === 'two-sample' ? '分組欄位 (X)' : '樣本 2 欄位'}
              </label>
              <select
                value={column2}
                onChange={(e) => setColumn2(e.target.value)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
              >
                <option value="">選擇欄位...</option>
                {dataset.headers.map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          )}
        </div>
        
        <button
          onClick={runTest}
          disabled={loading || !column1}
          className="mt-4 px-6 py-3 bg-emerald-600 text-white font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '計算中...' : '執行檢定'}
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
          <h3 className="font-bold text-slate-900 mb-4">📊 檢定結果: {result.test_type_zh}</h3>
          
          {/* Significance Result */}
          <div className={`p-4 rounded-lg mb-6 ${result.significant_005 ? 'bg-green-50 border border-green-200' : 'bg-slate-50 border border-slate-200'}`}>
            <div className="flex items-center gap-2 mb-2">
              {result.significant_005 ? (
                <CheckCircle className="text-green-600" size={24} />
              ) : (
                <XCircle className="text-slate-400" size={24} />
              )}
              <span className={`font-bold text-lg ${result.significant_005 ? 'text-green-700' : 'text-slate-600'}`}>
                {result.significant_005 ? '顯著差異 (p < 0.05)' : '無顯著差異 (p ≥ 0.05)'}
              </span>
            </div>
            <p className="text-sm text-slate-600">
              {result.significant_005 
                ? '有足夠證據拒絕虛無假設 (H₀)' 
                : '沒有足夠證據拒絕虛無假設 (H₀)'}
            </p>
          </div>
          
          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">T 統計量</div>
              <div className="text-xl font-bold text-slate-800">{result.t_statistic.toFixed(4)}</div>
            </div>
            <div className={`rounded-lg p-4 text-center ${result.significant_005 ? 'bg-green-50' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">P 值</div>
              <div className={`text-xl font-bold ${result.significant_005 ? 'text-green-700' : 'text-slate-800'}`}>
                {result.p_value < 0.0001 ? '< 0.0001' : result.p_value.toFixed(4)}
              </div>
            </div>
            {result.sample_mean !== undefined && (
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">樣本平均</div>
                <div className="text-xl font-bold text-slate-800">{result.sample_mean.toFixed(4)}</div>
              </div>
            )}
            {result.mean_difference !== undefined && (
              <div className="bg-amber-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">平均差異</div>
                <div className="text-xl font-bold text-amber-700">{result.mean_difference.toFixed(4)}</div>
              </div>
            )}
            {result.sample_size !== undefined && (
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">樣本大小</div>
                <div className="text-xl font-bold text-slate-800">{result.sample_size}</div>
              </div>
            )}
            {result.paired_size !== undefined && (
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">配對數</div>
                <div className="text-xl font-bold text-slate-800">{result.paired_size}</div>
              </div>
            )}
          </div>
          
          {/* Two-sample group details */}
          {result.group1_name && result.group2_name && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="text-sm font-medium text-blue-800 mb-2">組別: {result.group1_name}</div>
                <div className="text-xs text-slate-500">平均: <span className="font-mono">{result.sample1_mean.toFixed(4)}</span></div>
                <div className="text-xs text-slate-500">標準差: <span className="font-mono">{result.sample1_std.toFixed(4)}</span></div>
                <div className="text-xs text-slate-500">樣本數: <span className="font-mono">{result.sample1_size}</span></div>
              </div>
              <div className="bg-purple-50 rounded-lg p-4">
                <div className="text-sm font-medium text-purple-800 mb-2">組別: {result.group2_name}</div>
                <div className="text-xs text-slate-500">平均: <span className="font-mono">{result.sample2_mean.toFixed(4)}</span></div>
                <div className="text-xs text-slate-500">標準差: <span className="font-mono">{result.sample2_std.toFixed(4)}</span></div>
                <div className="text-xs text-slate-500">樣本數: <span className="font-mono">{result.sample2_size}</span></div>
              </div>
            </div>
          )}
          
          {/* Significance Levels */}
          <div className="mt-4 flex gap-4 text-sm">
            <div className={`px-3 py-1 rounded-full ${result.significant_005 ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'}`}>
              α = 0.05: {result.significant_005 ? '✓ 顯著' : '✗ 不顯著'}
            </div>
            <div className={`px-3 py-1 rounded-full ${result.significant_001 ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'}`}>
              α = 0.01: {result.significant_001 ? '✓ 顯著' : '✗ 不顯著'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HypothesisTesting;
