import React, { useState } from 'react';
import { Play, AlertCircle, Cloud } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface WordData {
  text: string;
  count: number;
  size: number;
}

interface WordCloudResult {
  column: string;
  total_words: number;
  unique_words: number;
  words: WordData[];
}

const WordCloud: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [column, setColumn] = useState<string>('');
  const [maxWords, setMaxWords] = useState<number>(50);
  const [minFrequency, setMinFrequency] = useState<number>(1);
  const [result, setResult] = useState<WordCloudResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runAnalysis = async () => {
    if (!column) {
      setError('Please select a column.');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      const response = await fetch(`${apiUrl}/word-cloud`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          column,
          max_words: maxWords,
          min_frequency: minFrequency
        })
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Analysis failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Failed to generate word cloud');
    } finally {
      setLoading(false);
    }
  };

  // Generate random colors for words
  const getRandomColor = (index: number) => {
    const colors = [
      '#2563eb', '#7c3aed', '#059669', '#dc2626', '#d97706',
      '#0891b2', '#4f46e5', '#16a34a', '#ea580c', '#0d9488',
      '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899', '#84cc16'
    ];
    return colors[index % colors.length];
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <Cloud size={20} className="text-blue-500" />
          文字雲設定
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">選擇文字欄位</label>
            <select
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">選擇欄位...</option>
              {dataset.headers.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">最多顯示字數</label>
            <input
              type="number"
              value={maxWords}
              onChange={(e) => setMaxWords(parseInt(e.target.value) || 50)}
              min={10}
              max={200}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">最低出現次數</label>
            <input
              type="number"
              value={minFrequency}
              onChange={(e) => setMinFrequency(parseInt(e.target.value) || 1)}
              min={1}
              max={10}
              className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        
        <button
          onClick={runAnalysis}
          disabled={loading || !column}
          className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '生成中...' : '生成文字雲'}
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
          {/* Stats */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-slate-900">📊 統計資訊</h3>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">總字詞數</div>
                <div className="text-xl font-bold text-slate-800">{result.total_words.toLocaleString()}</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">不重複字詞</div>
                <div className="text-xl font-bold text-slate-800">{result.unique_words.toLocaleString()}</div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">顯示字詞</div>
                <div className="text-xl font-bold text-slate-800">{result.words.length}</div>
              </div>
            </div>
          </div>
          
          {/* Word Cloud Visualization */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">☁️ 文字雲</h3>
            <div className="min-h-[300px] p-6 bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg flex flex-wrap justify-center items-center gap-2">
              {result.words.map((word, index) => (
                <span
                  key={index}
                  style={{
                    fontSize: `${word.size}px`,
                    color: getRandomColor(index),
                    fontWeight: word.size > 30 ? 'bold' : 'normal',
                    opacity: 0.7 + (word.count / result.words[0].count) * 0.3,
                    padding: '2px 4px',
                    cursor: 'pointer',
                    transition: 'transform 0.2s, opacity 0.2s'
                  }}
                  className="hover:opacity-100 hover:scale-110"
                  title={`${word.text}: ${word.count} 次`}
                >
                  {word.text}
                </span>
              ))}
            </div>
          </div>
          
          {/* Top Words Table */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📋 高頻字詞排行</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm border border-slate-200">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-4 py-2 border-b text-left">排名</th>
                    <th className="px-4 py-2 border-b text-left">字詞</th>
                    <th className="px-4 py-2 border-b text-right">出現次數</th>
                    <th className="px-4 py-2 border-b text-left">頻率條</th>
                  </tr>
                </thead>
                <tbody>
                  {result.words.slice(0, 20).map((word, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="px-4 py-2 border-b text-slate-500">{idx + 1}</td>
                      <td className="px-4 py-2 border-b font-medium" style={{color: getRandomColor(idx)}}>{word.text}</td>
                      <td className="px-4 py-2 border-b text-right font-mono">{word.count}</td>
                      <td className="px-4 py-2 border-b">
                        <div className="w-full bg-slate-200 rounded-full h-2">
                          <div
                            className="h-2 rounded-full"
                            style={{
                              width: `${(word.count / result.words[0].count) * 100}%`,
                              backgroundColor: getRandomColor(idx)
                            }}
                          />
                        </div>
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

export default WordCloud;
