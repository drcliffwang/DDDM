import React from 'react';
import { BarChart3, AlertCircle, Database, TrendingUp } from 'lucide-react';

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
  missing_values: {
    [key: string]: {
      count: number;
      percentage: number;
    };
  };
}

interface Props {
  statistics: StatisticsData | null;
  isLoading: boolean;
}

const EDAStats: React.FC<Props> = ({ statistics, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <span className="ml-3 text-slate-600">Calculating statistics...</span>
        </div>
      </div>
    );
  }

  if (!statistics) return null;

  const { summary, numeric_stats, missing_values } = statistics;
  const numericColumns = Object.keys(numeric_stats);
  
  // Calculate overall data quality
  const totalMissing = Object.values(missing_values).reduce((sum, val) => sum + val.count, 0);
  const dataQuality = ((1 - (totalMissing / (summary.total_rows * summary.total_columns))) * 100).toFixed(1);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-purple-500">
      <div className="p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <BarChart3 className="text-purple-600" size={24} />
              探索性數據分析 (EDA)
            </h2>
            <p className="mt-1 text-slate-500">Descriptive statistics and data quality overview</p>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
            <div className="flex items-center gap-2 text-blue-600 mb-1">
              <Database size={16} />
              <span className="text-xs font-semibold uppercase">Total Rows</span>
            </div>
            <p className="text-2xl font-bold text-blue-900">{summary.total_rows.toLocaleString()}</p>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
            <div className="flex items-center gap-2 text-green-600 mb-1">
              <TrendingUp size={16} />
              <span className="text-xs font-semibold uppercase">Columns</span>
            </div>
            <p className="text-2xl font-bold text-green-900">{summary.total_columns}</p>
            <p className="text-xs text-green-700 mt-1">
              {summary.numeric_columns} numeric, {summary.categorical_columns} categorical
            </p>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
            <div className="flex items-center gap-2 text-purple-600 mb-1">
              <AlertCircle size={16} />
              <span className="text-xs font-semibold uppercase">Missing Values</span>
            </div>
            <p className="text-2xl font-bold text-purple-900">{totalMissing.toLocaleString()}</p>
          </div>

          <div className="bg-gradient-to-br from-amber-50 to-amber-100 p-4 rounded-lg border border-amber-200">
            <div className="flex items-center gap-2 text-amber-600 mb-1">
              <BarChart3 size={16} />
              <span className="text-xs font-semibold uppercase">Data Quality</span>
            </div>
            <p className="text-2xl font-bold text-amber-900">{dataQuality}%</p>
          </div>
        </div>

        {/* Numeric Statistics Table */}
        {numericColumns.length > 0 && (
          <div className="mb-6">
            <h3 className="text-lg font-bold text-slate-900 mb-3">Numeric Column Statistics</h3>
            <div className="overflow-auto max-h-[400px] border border-slate-200 rounded-lg">
              <table className="min-w-full divide-y divide-slate-200">
                <thead className="bg-slate-50 sticky top-0 z-10">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-bold text-slate-500 uppercase">Column</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Count</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Mean</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Std</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Min</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Q1</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Median</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Q3</th>
                    <th className="px-4 py-3 text-right text-xs font-bold text-slate-500 uppercase">Max</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-slate-200">
                  {numericColumns.map((col, idx) => {
                    const stats = numeric_stats[col];
                    return (
                      <tr key={col} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                        <td className="px-4 py-3 text-sm font-medium text-slate-900">{col}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.count}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.mean.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.std.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.min.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.q1.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.median.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.q3.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-slate-700 text-right">{stats.max.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Missing Values */}
        {Object.keys(missing_values).some(col => missing_values[col].count > 0) && (
          <div>
            <h3 className="text-lg font-bold text-slate-900 mb-3">Missing Values by Column</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(missing_values)
                .filter(([_, val]) => val.count > 0)
                .map(([col, val]) => (
                  <div key={col} className="bg-red-50 border border-red-200 rounded-lg p-3">
                    <p className="text-sm font-semibold text-slate-900 truncate">{col}</p>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-xs text-slate-600">{val.count} missing</span>
                      <span className="text-xs font-bold text-red-600">{val.percentage}%</span>
                    </div>
                    <div className="mt-2 bg-red-200 rounded-full h-2">
                      <div 
                        className="bg-red-500 h-2 rounded-full transition-all"
                        style={{ width: `${val.percentage}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EDAStats;
