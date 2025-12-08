import React, { useState } from 'react';
import { LineChart, Activity } from 'lucide-react';
import Plot from 'react-plotly.js';
import {
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie
} from 'recharts';

interface StatisticsData {
  summary: {
    total_rows: number;
    total_columns: number;
    numeric_columns: number;
    categorical_columns: number;
  };
  numeric_stats: {
    [key: string]: {
      count: number;
      mean: number;
      std: number;
      min: number;
      q1: number;
      median: number;
      q3: number;
      max: number;
    };
  };
  categorical_stats: {
    [key: string]: Array<{
      value: string;
      count: number;
    }>;
  };
  correlation_matrix: {
    columns: string[];
    values: number[][];
  };
}

interface Props {
  statistics: StatisticsData | null;
  isLoading: boolean;
  dataset: { fileName: string; headers: string[]; rows: Array<{[key: string]: any}> } | null;
}

const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444', '#14b8a6'];

const DataVisualization: React.FC<Props> = ({ statistics, isLoading, dataset }) => {
  const [selectedNumericCol, setSelectedNumericCol] = useState<string>('');
  const [selectedCategoricalCol, setSelectedCategoricalCol] = useState<string>('');

  // Extract data safely
  const numericColumns = statistics ? Object.keys(statistics.numeric_stats) : [];
  const categoricalColumns = statistics ? Object.keys(statistics.categorical_stats) : [];

  // Use useEffect to set default selections (must be called before any early returns)
  React.useEffect(() => {
    if (!selectedNumericCol && numericColumns.length > 0) {
      setSelectedNumericCol(numericColumns[0]);
    }
    if (!selectedCategoricalCol && categoricalColumns.length > 0) {
      setSelectedCategoricalCol(categoricalColumns[0]);
    }
  }, [numericColumns.length, categoricalColumns.length, selectedNumericCol, selectedCategoricalCol]);

  // Early returns AFTER all hooks
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <span className="ml-3 text-slate-600">Generating visualizations...</span>
        </div>
      </div>
    );
  }

  if (!statistics) return null;

  const { categorical_stats, correlation_matrix } = statistics;

  // Prepare pie chart data for categorical columns
  const getPieChartData = (col: string) => {
    const data = categorical_stats[col] || [];
    const total = data.reduce((sum, item) => sum + item.count, 0);
    
    return data.map((item, index) => ({
      name: item.value,
      value: item.count,
      percentage: ((item.count / total) * 100).toFixed(1),
      fill: COLORS[index % COLORS.length]
    }));
  };




  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-indigo-500">
      <div className="p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <Activity className="text-indigo-600" size={24} />
              數據視覺化 (Data Visualization)
            </h2>
            <p className="mt-1 text-slate-500">Interactive charts and distributions</p>
          </div>
        </div>

        {/* Debug info - showing what columns were detected */}
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm">
          <div className="font-semibold text-blue-900 mb-2">📊 Dataset Detection:</div>
          <div className="text-blue-700">
            • Categorical columns detected: <span className="font-mono font-bold">{categoricalColumns.length}</span>
            {categoricalColumns.length > 0 && ` (${categoricalColumns.join(', ')})`}
          </div>
          {categoricalColumns.length === 0 && (
            <div className="mt-2 text-blue-600 text-xs">
              ℹ️ No categorical columns found. All columns appear to be numeric. Pie chart requires text/categorical data.
            </div>
          )}
        </div>

        <div className="space-y-8">
          
          {/* Numeric Distribution - Violin + Box Plot */}
          {numericColumns.length > 0 && selectedNumericCol && (
            <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                  <Activity size={20} className="text-teal-600" />
                  分佈詳情 Distribution (Violin + Box)
                </h3>
                <select
                  value={selectedNumericCol}
                  onChange={(e) => setSelectedNumericCol(e.target.value)}
                  className="px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-teal-500 focus:border-teal-500"
                >
                  {numericColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              
              {/* Plotly Violin + Box Plot */}
              <Plot
                data={[
                  {
                    type: 'violin',
                    y: dataset && dataset.rows ? 
                       // Extract REAL values from the dataset
                       dataset.rows
                         .map(row => row[selectedNumericCol])
                         .filter(val => val !== null && val !== undefined && !isNaN(Number(val)))
                         .map(val => Number(val))
                       : [],
                    name: selectedNumericCol,
                    box: {
                      visible: true,
                    },
                    meanline: {
                      visible: true
                    },
                    fillcolor: 'rgba(99, 102, 241, 0.3)',
                    line: {
                      color: 'rgb(99, 102, 241)'
                    },
                    marker: {
                      size: 3,
                      color: 'rgba(99, 102, 241, 0.6)'
                    },
                    points: 'all',
                    jitter: 0.3,
                    hovertemplate: '<b>Value</b>: %{y:.2f}<extra></extra>'
                  }
                ]}
                layout={{
                  title: {
                    text: `${selectedNumericCol} - Distribution Analysis`
                  },
                  yaxis: {
                    title: selectedNumericCol,
                    gridcolor: '#e2e8f0',
                    zeroline: false
                  },
                  xaxis: {
                    showticklabels: false
                  },
                  height: 400,
                  margin: { l: 60, r: 30, t: 50, b: 40 },
                  plot_bgcolor: '#ffffff',
                  paper_bgcolor: '#f8fafc',
                  font: { family: 'Inter, sans-serif', size: 11 },
                  showlegend: false
                }}
                config={{
                  responsive: true,
                  displayModeBar: false
                }}
                style={{ width: '100%' }}
              />
              
              <div className="mt-3 text-xs text-slate-500 text-center">
                Violin plot showing distribution density + box plot showing quartiles (Q1, Median, Q3)
              </div>
            </div>
          )}


          {/* Categorical Distribution Pie Chart */}
          {categoricalColumns.length > 0 && (
            <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                  <LineChart size={20} className="text-purple-600" />
                  類別佔比 Category Distribution
                </h3>
                <select
                  value={selectedCategoricalCol}
                  onChange={(e) => setSelectedCategoricalCol(e.target.value)}
                  className="px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                >
                  {categoricalColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              
              <ResponsiveContainer width="100%" height={350}>
                <PieChart>
                  <Pie
                    data={getPieChartData(selectedCategoricalCol)}
                    cx="50%"
                    cy="50%"
                    innerRadius={80}
                    outerRadius={120}
                    paddingAngle={2}
                    dataKey="value"
                    label={(entry) => `${entry.name}: ${entry.percentage}%`}
                    labelLine={{ stroke: '#64748b', strokeWidth: 1 }}
                  >
                    {getPieChartData(selectedCategoricalCol).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                    formatter={(value: any, name: string, props: any) => [
                      `${value} (${props.payload.percentage}%)`,
                      name
                    ]}
                  />
                  <Legend 
                    verticalAlign="bottom" 
                    height={36}
                    formatter={(value, entry: any) => `${value} (${entry.payload.value})`}
                  />
                </PieChart>
              </ResponsiveContainer>
              
              <div className="mt-3 text-xs text-slate-500 text-center">
                Donut chart showing category distribution with percentages
              </div>
            </div>
          )}

          {/* Correlation Matrix Visualization */}
          {correlation_matrix.columns && correlation_matrix.columns.length > 1 && (
            <div className="bg-slate-50 rounded-lg p-6 border border-slate-200">
              <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                <Activity size={20} className="text-green-600" />
                Correlation Matrix
              </h3>
              
              <div className="overflow-x-auto">
                <table className="min-w-full border border-slate-300">
                  <thead>
                    <tr>
                      <th className="border border-slate-300 bg-slate-100 px-3 py-2 text-xs font-bold"></th>
                      {correlation_matrix.columns.map(col => (
                        <th key={col} className="border border-slate-300 bg-slate-100 px-3 py-2 text-xs font-bold">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {correlation_matrix.columns.map((row, i) => (
                      <tr key={row}>
                        <td className="border border-slate-300 bg-slate-100 px-3 py-2 text-xs font-bold">
                          {row}
                        </td>
                        {correlation_matrix.columns.map((col, j) => {
                          const value = correlation_matrix.values[i][j];
                          const intensity = Math.abs(value);
                          const isPositive = value >= 0;
                          const bgColor = isPositive 
                            ? `rgba(34, 197, 94, ${intensity})` 
                            : `rgba(239, 68, 68, ${intensity})`;
                          
                          return (
                            <td
                              key={col}
                              className="border border-slate-300 px-3 py-2 text-center text-xs font-mono"
                              style={{ backgroundColor: bgColor }}
                            >
                              {value.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-4 flex items-center justify-center gap-6 text-xs text-slate-600">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-green-500 rounded"></div>
                  <span>Positive Correlation</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-red-500 rounded"></div>
                  <span>Negative Correlation</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DataVisualization;
