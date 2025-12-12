import React, { useState } from 'react';
import { Play, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface ChiSquareResult {
  chi_square_statistic: number;
  p_value: number;
  degrees_of_freedom: number;
  cramers_v: number;
  significant_005: boolean;
  significant_001: boolean;
  observed_table: Record<string, Record<string, number>>;
  expected_table: Record<string, Record<string, number>>;
  row_categories: string[];
  col_categories: string[];
  table_shape: number[];
}

const ChiSquareTest: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [column1, setColumn1] = useState<string>('');
  const [column2, setColumn2] = useState<string>('');
  const [result, setResult] = useState<ChiSquareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showExpected, setShowExpected] = useState(false);

  const runTest = async () => {
    if (!column1 || !column2) {
      setError('Please select both columns.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/chi-square-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          column1,
          column2
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Test failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to run Chi-Square test');
    } finally {
      setLoading(false);
    }
  };

  const getCramersVInterpretation = (v: number) => {
    if (v < 0.1) return { text: '可忽略', color: 'text-slate-500' };
    if (v < 0.3) return { text: '弱', color: 'text-amber-600' };
    if (v < 0.5) return { text: '中等', color: 'text-blue-600' };
    return { text: '強', color: 'text-green-600' };
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4">📐 卡方檢定設定 (Chi-Square Test)</h3>
        <p className="text-sm text-slate-500 mb-4">檢驗兩個類別變數之間是否存在關聯性</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">類別變數 1</label>
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
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">類別變數 2</label>
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
        </div>
        
        <button
          onClick={runTest}
          disabled={loading || !column1 || !column2}
          className="mt-4 px-6 py-3 bg-emerald-600 text-white font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '計算中...' : '執行卡方檢定'}
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
          <h3 className="font-bold text-slate-900 mb-4">📊 卡方檢定結果</h3>
          
          {/* Significance Result */}
          <div className={`p-4 rounded-lg mb-6 ${result.significant_005 ? 'bg-green-50 border border-green-200' : 'bg-slate-50 border border-slate-200'}`}>
            <div className="flex items-center gap-2 mb-2">
              {result.significant_005 ? (
                <CheckCircle className="text-green-600" size={24} />
              ) : (
                <XCircle className="text-slate-400" size={24} />
              )}
              <span className={`font-bold text-lg ${result.significant_005 ? 'text-green-700' : 'text-slate-600'}`}>
                {result.significant_005 ? '變數間存在顯著關聯 (p < 0.05)' : '變數間無顯著關聯 (p ≥ 0.05)'}
              </span>
            </div>
            <p className="text-sm text-slate-600">
              {result.significant_005 
                ? '兩個類別變數之間存在統計顯著的關聯性' 
                : '沒有足夠證據表明兩個類別變數之間有關聯'}
            </p>
          </div>
          
          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">χ² 統計量</div>
              <div className="text-xl font-bold text-slate-800">{result.chi_square_statistic.toFixed(4)}</div>
            </div>
            <div className={`rounded-lg p-4 text-center ${result.significant_005 ? 'bg-green-50' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">P 值</div>
              <div className={`text-xl font-bold ${result.significant_005 ? 'text-green-700' : 'text-slate-800'}`}>
                {result.p_value < 0.0001 ? '< 0.0001' : result.p_value.toFixed(4)}
              </div>
            </div>
            <div className="bg-slate-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">自由度 (df)</div>
              <div className="text-xl font-bold text-slate-800">{result.degrees_of_freedom}</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">Cramér's V</div>
              <div className={`text-xl font-bold ${getCramersVInterpretation(result.cramers_v).color}`}>
                {result.cramers_v.toFixed(4)}
              </div>
              <div className="text-xs text-slate-400">{getCramersVInterpretation(result.cramers_v).text}關聯</div>
            </div>
          </div>
          
          {/* Contingency Table Toggle */}
          <div className="mb-4">
            <div className="flex gap-2">
              <button
                onClick={() => setShowExpected(false)}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${!showExpected ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'}`}
              >
                觀察值表
              </button>
              <button
                onClick={() => setShowExpected(true)}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${showExpected ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'}`}
              >
                期望值表
              </button>
            </div>
          </div>
          
          {/* Contingency Table */}
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm border border-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2 border-b border-r border-slate-200">{column1} \ {column2}</th>
                  {result.col_categories.map((col) => (
                    <th key={String(col)} className="px-4 py-2 border-b border-slate-200 text-center">{String(col)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.row_categories.map((row) => (
                  <tr key={String(row)} className="hover:bg-slate-50">
                    <td className="px-4 py-2 border-r border-b border-slate-200 font-medium bg-slate-50">{String(row)}</td>
                    {result.col_categories.map((col) => {
                      const table = showExpected ? result.expected_table : result.observed_table;
                      const value = table[String(row)]?.[String(col)] ?? 0;
                      return (
                        <td key={String(col)} className="px-4 py-2 border-b border-slate-200 text-center font-mono">
                          {showExpected ? value.toFixed(2) : value}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
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

export default ChiSquareTest;
