import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from 'recharts';
import { Play, AlertCircle, TrendingUp } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
  apiUrl: string;
}

interface SimpleRegressionResult {
  slope: number;
  intercept: number;
  r_squared: number;
  adj_r_squared: number;
  correlation: number;
  p_value: number;
  t_statistic: number;
  se_slope: number;
  n_observations: number;
  significant_005: boolean;
  equation: string;
  scatter_data: Array<{x: number; y: number}>;
  line_start: {x: number; y: number};
  line_end: {x: number; y: number};
}

interface MultipleRegressionResult {
  intercept: number;
  coefficients: Array<{variable: string; coefficient: number}>;
  r_squared: number;
  adj_r_squared: number;
  f_statistic: number;
  f_p_value: number;
  n_observations: number;
  n_predictors: number;
  significant_005: boolean;
}

const RegressionAnalysis: React.FC<Props> = ({ dataset, apiUrl }) => {
  const [regressionType, setRegressionType] = useState<'simple' | 'multiple'>('simple');
  const [xColumn, setXColumn] = useState<string>('');
  const [xColumns, setXColumns] = useState<string[]>([]);
  const [yColumn, setYColumn] = useState<string>('');
  const [simpleResult, setSimpleResult] = useState<SimpleRegressionResult | null>(null);
  const [multiResult, setMultiResult] = useState<MultipleRegressionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // Prediction states
  const [simplePredictX, setSimplePredictX] = useState<string>('');
  const [simplePredictY, setSimplePredictY] = useState<number | null>(null);
  const [multiPredictInputs, setMultiPredictInputs] = useState<Record<string, string>>({});
  const [multiPredictY, setMultiPredictY] = useState<number | null>(null);

  const toggleXColumn = (col: string) => {
    if (xColumns.includes(col)) {
      setXColumns(xColumns.filter(c => c !== col));
    } else {
      setXColumns([...xColumns, col]);
    }
  };

  const runAnalysis = async () => {
    setLoading(true);
    setError('');
    setSimpleResult(null);
    setMultiResult(null);
    
    try {
      if (regressionType === 'simple') {
        if (!xColumn || !yColumn) {
          throw new Error('Please select both X and Y columns.');
        }
        const response = await fetch(`${apiUrl}/regression-analysis`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: dataset.rows, x_column: xColumn, y_column: yColumn })
        });
        if (!response.ok) throw new Error((await response.json()).detail || 'Analysis failed');
        setSimpleResult(await response.json());
      } else {
        if (xColumns.length === 0 || !yColumn) {
          throw new Error('Please select at least one X column and Y column.');
        }
        const response = await fetch(`${apiUrl}/multiple-regression`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: dataset.rows, x_columns: xColumns, y_column: yColumn })
        });
        if (!response.ok) throw new Error((await response.json()).detail || 'Analysis failed');
        setMultiResult(await response.json());
      }
    } catch (e: any) {
      setError(e.message || 'Failed to run regression analysis');
    } finally {
      setLoading(false);
    }
  };

  const getR2Interpretation = (r2: number) => {
    if (r2 >= 0.9) return { text: '非常強', color: 'text-green-600' };
    if (r2 >= 0.7) return { text: '強', color: 'text-blue-600' };
    if (r2 >= 0.5) return { text: '中等', color: 'text-amber-600' };
    if (r2 >= 0.3) return { text: '弱', color: 'text-orange-600' };
    return { text: '非常弱', color: 'text-red-600' };
  };

  // Simple regression prediction
  const predictSimple = () => {
    if (!simpleResult || !simplePredictX) return;
    const x = parseFloat(simplePredictX);
    if (isNaN(x)) return;
    const y = simpleResult.slope * x + simpleResult.intercept;
    setSimplePredictY(y);
  };

  // Multiple regression prediction
  const predictMultiple = () => {
    if (!multiResult) return;
    let y = multiResult.intercept;
    for (const c of multiResult.coefficients) {
      const inputVal = parseFloat(multiPredictInputs[c.variable] || '0');
      if (isNaN(inputVal)) return;
      y += c.coefficient * inputVal;
    }
    setMultiPredictY(y);
  };

  return (
    <div className="mt-4 space-y-6">
      {/* Configuration */}
      <div className="bg-white rounded-xl p-6 border border-slate-200">
        <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
          <TrendingUp size={20} className="text-emerald-600" />
          迴歸分析設定
        </h3>
        
        {/* Regression Type Toggle */}
        <div className="mb-4">
          <div className="flex gap-2">
            <button
              onClick={() => setRegressionType('simple')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                regressionType === 'simple' ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              簡單迴歸
            </button>
            <button
              onClick={() => setRegressionType('multiple')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                regressionType === 'multiple' ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              多元迴歸
            </button>
          </div>
        </div>
        
        {regressionType === 'simple' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">自變數 X</label>
              <select
                value={xColumn}
                onChange={(e) => setXColumn(e.target.value)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
              >
                <option value="">選擇欄位...</option>
                {dataset.headers.filter(h => h !== yColumn).map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">應變數 Y</label>
              <select
                value={yColumn}
                onChange={(e) => setYColumn(e.target.value)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
              >
                <option value="">選擇欄位...</option>
                {dataset.headers.map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">應變數 Y</label>
              <select
                value={yColumn}
                onChange={(e) => setYColumn(e.target.value)}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
              >
                <option value="">選擇欄位...</option>
                {dataset.headers.map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">自變數 Xs (可多選)</label>
              <div className="flex flex-wrap gap-2">
                {dataset.headers.filter(h => h !== yColumn).map((h) => (
                  <button
                    key={h}
                    onClick={() => toggleXColumn(h)}
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                      xColumns.includes(h)
                        ? 'bg-emerald-600 text-white'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {h}
                  </button>
                ))}
              </div>
              {xColumns.length > 0 && (
                <p className="mt-2 text-sm text-emerald-600">已選: {xColumns.join(', ')}</p>
              )}
            </div>
          </div>
        )}
        
        <button
          onClick={runAnalysis}
          disabled={loading || (regressionType === 'simple' ? !xColumn || !yColumn : xColumns.length === 0 || !yColumn)}
          className="mt-4 px-6 py-3 bg-emerald-600 text-white font-medium rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play size={18} />
          {loading ? '分析中...' : '執行迴歸分析'}
        </button>
        
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-700">
            <AlertCircle size={18} />
            {error}
          </div>
        )}
      </div>
      
      {/* Simple Regression Results */}
      {simpleResult && (
        <>
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📐 簡單迴歸方程式</h3>
            <div className="bg-slate-50 rounded-lg p-4 mb-6 text-center">
              <code className="text-xl font-mono text-slate-800">{simpleResult.equation}</code>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`rounded-lg p-4 text-center ${simpleResult.r_squared >= 0.5 ? 'bg-green-50' : 'bg-slate-50'}`}>
                <div className="text-xs text-slate-500">R²</div>
                <div className={`text-xl font-bold ${getR2Interpretation(simpleResult.r_squared).color}`}>
                  {(simpleResult.r_squared * 100).toFixed(2)}%
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">相關係數 (r)</div>
                <div className="text-xl font-bold text-slate-800">{simpleResult.correlation.toFixed(4)}</div>
              </div>
              <div className={`rounded-lg p-4 text-center ${simpleResult.significant_005 ? 'bg-green-50' : 'bg-slate-50'}`}>
                <div className="text-xs text-slate-500">P 值</div>
                <div className={`text-xl font-bold ${simpleResult.significant_005 ? 'text-green-700' : 'text-slate-800'}`}>
                  {simpleResult.p_value < 0.0001 ? '< 0.0001' : simpleResult.p_value.toFixed(4)}
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4 text-center">
                <div className="text-xs text-slate-500">觀察數</div>
                <div className="text-xl font-bold text-slate-800">{simpleResult.n_observations}</div>
              </div>
            </div>
          </div>
          
          {/* Scatter Plot */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">📈 散點圖與迴歸線</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" label={{ value: xColumn, position: 'insideBottomRight', offset: -10 }} />
                  <YAxis type="number" dataKey="y" label={{ value: yColumn, angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <Scatter data={simpleResult.scatter_data} fill="#10b981" />
                  <ReferenceLine
                    segment={[
                      { x: simpleResult.line_start.x, y: simpleResult.line_start.y },
                      { x: simpleResult.line_end.x, y: simpleResult.line_end.y }
                    ]}
                    stroke="#ef4444"
                    strokeWidth={2}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Simple Regression Prediction */}
          <div className="bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4">🔮 預測 Y 值</h3>
            <div className="flex gap-4 items-end">
              <div className="flex-1">
                <label className="block text-sm font-medium text-slate-700 mb-2">輸入 {xColumn} 的值</label>
                <input
                  type="number"
                  value={simplePredictX}
                  onChange={(e) => setSimplePredictX(e.target.value)}
                  className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500"
                  placeholder="輸入 X 值..."
                />
              </div>
              <button
                onClick={predictSimple}
                disabled={!simplePredictX}
                className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                計算預測值
              </button>
            </div>
            {simplePredictY !== null && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <span className="text-slate-600">當 {xColumn} = </span>
                <span className="font-bold text-slate-800">{simplePredictX}</span>
                <span className="text-slate-600"> 時，預測 {yColumn} = </span>
                <span className="text-2xl font-bold text-blue-700">{simplePredictY.toFixed(4)}</span>
              </div>
            )}
          </div>
        </>
      )}
      
      {/* Multiple Regression Results */}
      {multiResult && (
        <div className="bg-white rounded-xl p-6 border border-slate-200">
          <h3 className="font-bold text-slate-900 mb-4">📐 多元迴歸結果</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className={`rounded-lg p-4 text-center ${multiResult.r_squared >= 0.5 ? 'bg-green-50' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">R²</div>
              <div className={`text-xl font-bold ${getR2Interpretation(multiResult.r_squared).color}`}>
                {(multiResult.r_squared * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-slate-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">調整後 R²</div>
              <div className="text-xl font-bold text-slate-800">{(multiResult.adj_r_squared * 100).toFixed(2)}%</div>
            </div>
            <div className={`rounded-lg p-4 text-center ${multiResult.significant_005 ? 'bg-green-50' : 'bg-slate-50'}`}>
              <div className="text-xs text-slate-500">F 統計量 / P 值</div>
              <div className={`text-lg font-bold ${multiResult.significant_005 ? 'text-green-700' : 'text-slate-800'}`}>
                {multiResult.f_statistic.toFixed(2)} / {multiResult.f_p_value < 0.0001 ? '<0.0001' : multiResult.f_p_value.toFixed(4)}
              </div>
            </div>
            <div className="bg-slate-50 rounded-lg p-4 text-center">
              <div className="text-xs text-slate-500">觀察數 / 預測變數</div>
              <div className="text-xl font-bold text-slate-800">{multiResult.n_observations} / {multiResult.n_predictors}</div>
            </div>
          </div>
          
          <h4 className="font-semibold text-slate-700 mb-3">迴歸係數</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm border border-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2 border-b text-left">變數</th>
                  <th className="px-4 py-2 border-b text-right">係數 (β)</th>
                </tr>
              </thead>
              <tbody>
                <tr className="bg-blue-50">
                  <td className="px-4 py-2 border-b font-medium">截距</td>
                  <td className="px-4 py-2 border-b text-right font-mono">{multiResult.intercept.toFixed(4)}</td>
                </tr>
                {multiResult.coefficients.map((c, idx) => (
                  <tr key={idx} className="hover:bg-slate-50">
                    <td className="px-4 py-2 border-b">{c.variable}</td>
                    <td className="px-4 py-2 border-b text-right font-mono">{c.coefficient.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Coefficients Bar Chart */}
          <div className="mt-6">
            <h4 className="font-semibold text-slate-700 mb-3">係數視覺化</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={multiResult.coefficients} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="variable" type="category" width={100} tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <Bar dataKey="coefficient" radius={[0, 4, 4, 0]}>
                    {multiResult.coefficients.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.coefficient >= 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 flex gap-4 text-sm text-slate-500">
              <span className="flex items-center gap-1"><span className="w-3 h-3 bg-emerald-500 rounded"></span> 正向影響</span>
              <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-500 rounded"></span> 負向影響</span>
            </div>
          </div>

          {/* Multiple Regression Prediction */}
          <div className="mt-6 bg-blue-50 rounded-lg p-6">
            <h4 className="font-semibold text-slate-700 mb-3">🔮 預測 Y 值</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
              {multiResult.coefficients.map((c) => (
                <div key={c.variable}>
                  <label className="block text-sm font-medium text-slate-600 mb-1">{c.variable}</label>
                  <input
                    type="number"
                    value={multiPredictInputs[c.variable] || ''}
                    onChange={(e) => setMultiPredictInputs({...multiPredictInputs, [c.variable]: e.target.value})}
                    className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
                    placeholder="輸入值..."
                  />
                </div>
              ))}
            </div>
            <button
              onClick={predictMultiple}
              className="px-6 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700"
            >
              計算預測值
            </button>
            {multiPredictY !== null && (
              <div className="mt-4 p-4 bg-white rounded-lg border border-blue-200">
                <span className="text-slate-600">預測 {yColumn} = </span>
                <span className="text-2xl font-bold text-blue-700">{multiPredictY.toFixed(4)}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default RegressionAnalysis;
