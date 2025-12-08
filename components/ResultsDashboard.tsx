import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ModelResult } from '../types';
import { Target, List, Layers, BarChart2 } from 'lucide-react';

interface Props {
  result: ModelResult;
}

const ResultsDashboard: React.FC<Props> = ({ result }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-blue-500">
      <div className="p-8">
        <div className="flex items-start justify-between mb-8">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-sm font-bold text-slate-600">3</div>
              Results Dashboard
            </h2>
            <p className="mt-1 text-slate-500 ml-10">Model performance metrics and feature importance</p>
          </div>
        </div>

        {/* Warning Banner for Missing Features */}
        {result.warning && (
          <div className="ml-10 mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-start gap-3">
              <div className="text-amber-600 mt-0.5">⚠️</div>
              <div className="flex-1">
                <h4 className="text-sm font-bold text-amber-900">Missing Features Handled</h4>
                <p className="text-sm text-amber-700 mt-1">{result.warning}</p>
                {result.features_skipped && result.features_skipped.length > 0 && (
                  <div className="mt-2 text-xs text-amber-600">
                    <span className="font-semibold">Skipped:</span> {result.features_skipped.join(', ')}
                  </div>
                )}
                {result.features_used && result.features_used.length > 0 && (
                  <div className="mt-1 text-xs text-amber-600">
                    <span className="font-semibold">Used:</span> {result.features_used.join(', ')}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="ml-10 grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Accuracy Card */}
          <div className="bg-slate-50 rounded-xl p-6 border border-slate-200 flex flex-col items-center justify-center text-center">
            <div className="p-3 bg-white rounded-full shadow-sm mb-4">
              <Target className={result.accuracy >= 0.7 ? "text-green-500" : result.accuracy >= 0.5 ? "text-amber-500" : "text-red-500"} size={32} />
            </div>
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wide">Model Accuracy</h3>
            <p className="text-4xl font-extrabold text-slate-900 mt-2">
              {(result.accuracy * 100).toFixed(1)}%
            </p>
            {/* Dynamic confidence based on accuracy */}
            {result.accuracy >= 0.8 ? (
              <span className="text-xs text-green-600 font-medium mt-1 px-2 py-1 bg-green-100 rounded-full">
                High Confidence
              </span>
            ) : result.accuracy >= 0.7 ? (
              <span className="text-xs text-green-600 font-medium mt-1 px-2 py-1 bg-green-100 rounded-full">
                Good
              </span>
            ) : result.accuracy >= 0.5 ? (
              <span className="text-xs text-amber-600 font-medium mt-1 px-2 py-1 bg-amber-100 rounded-full">
                Fair (Need Improvement)
              </span>
            ) : (
              <span className="text-xs text-red-600 font-medium mt-1 px-2 py-1 bg-red-100 rounded-full">
                Poor (Check Data/Features)
              </span>
            )}
            {/* Show test set size warning */}
            <div className="mt-3 text-xs text-slate-500">
              Test predictions: {result.confusionMatrix.flat().reduce((a, b) => a + b, 0)}
            </div>
          </div>

          {/* Feature Importance Chart */}
          <div className="lg:col-span-2 bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
            <h3 className="font-bold text-slate-900 mb-6 flex items-center gap-2">
              <BarChart2 size={18} className="text-indigo-600" />
              Feature Importance
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={result.featureImportance.slice(0, 5)} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    width={100} 
                    tick={{fill: '#475569', fontSize: 12}} 
                  />
                  <Tooltip 
                    cursor={{fill: '#f1f5f9'}}
                    contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {result.featureImportance.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff'][index % 5]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix (Simplified Visualization) */}
          <div className="lg:col-span-3 bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Layers size={18} className="text-orange-500" />
              Confusion Matrix
            </h3>
            <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
              <div className="bg-indigo-50 p-4 rounded-lg text-center border border-indigo-100">
                <span className="block text-xs text-indigo-600 font-bold uppercase">True Positive</span>
                <span className="block text-2xl font-bold text-indigo-900">{result.confusionMatrix[0][0]}</span>
              </div>
              <div className="bg-red-50 p-4 rounded-lg text-center border border-red-100">
                <span className="block text-xs text-red-600 font-bold uppercase">False Positive</span>
                <span className="block text-2xl font-bold text-red-900">{result.confusionMatrix[0][1]}</span>
              </div>
              <div className="bg-red-50 p-4 rounded-lg text-center border border-red-100">
                <span className="block text-xs text-red-600 font-bold uppercase">False Negative</span>
                <span className="block text-2xl font-bold text-red-900">{result.confusionMatrix[1][0]}</span>
              </div>
              <div className="bg-indigo-50 p-4 rounded-lg text-center border border-indigo-100">
                <span className="block text-xs text-indigo-600 font-bold uppercase">True Negative</span>
                <span className="block text-2xl font-bold text-indigo-900">{result.confusionMatrix[1][1]}</span>
              </div>
            </div>
          </div>

          {/* Precision, Recall, F1 Scores */}
          {(result.precision !== undefined || result.recall !== undefined || result.f1_score !== undefined) && (
            <div className="lg:col-span-3 bg-white rounded-xl p-6 border border-slate-200">
              <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                <List size={18} className="text-purple-500" />
                Classification Metrics
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Precision */}
                {result.precision !== undefined && (
                  <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-bold text-purple-900">Precision 精確率</span>
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        result.precision >= 0.8 ? 'bg-green-100 text-green-700' :
                        result.precision >= 0.6 ? 'bg-amber-100 text-amber-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {result.precision >= 0.8 ? 'High' : result.precision >= 0.6 ? 'Fair' : 'Low'}
                      </span>
                    </div>
                    <p className="text-3xl font-bold text-purple-900 mb-1">
                      {(result.precision * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-600">
                      預測為正類中，實際為正類的比例
                    </p>
                  </div>
                )}

                {/* Recall */}
                {result.recall !== undefined && (
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-bold text-blue-900">Recall 召回率</span>
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        result.recall >= 0.8 ? 'bg-green-100 text-green-700' :
                        result.recall >= 0.6 ? 'bg-amber-100 text-amber-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {result.recall >= 0.8 ? 'High' : result.recall >= 0.6 ? 'Fair' : 'Low'}
                      </span>
                    </div>
                    <p className="text-3xl font-bold text-blue-900 mb-1">
                      {(result.recall * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-600">
                      實際為正類中，被正確預測的比例
                    </p>
                  </div>
                )}

                {/* F1 Score */}
                {result.f1_score !== undefined && (
                  <div className="bg-emerald-50 p-4 rounded-lg border border-emerald-100">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-bold text-emerald-900">F1 Score F1分數</span>
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        result.f1_score >= 0.8 ? 'bg-green-100 text-green-700' :
                        result.f1_score >= 0.6 ? 'bg-amber-100 text-amber-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {result.f1_score >= 0.8 ? 'High' : result.f1_score >= 0.6 ? 'Fair' : 'Low'}
                      </span>
                    </div>
                    <p className="text-3xl font-bold text-emerald-900 mb-1">
                      {(result.f1_score * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-600">
                      精確率與召回率的調和平均數
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;