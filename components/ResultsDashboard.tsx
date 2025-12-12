import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line } from 'recharts';
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
          
          
          {/* Main Metric Card (Accuracy or R2) - Hide for Association Rules, PCA, FA, K-Means, Hierarchical */}
          {result.model_type !== 'association_rules' && result.model_type !== 'pca' && result.model_type !== 'factor-analysis' && result.model_type !== 'kmeans' && result.model_type !== 'hierarchical' && (
          <div className="bg-slate-50 rounded-xl p-6 border border-slate-200 flex flex-col items-center justify-center text-center">
            <div className="p-3 bg-white rounded-full shadow-sm mb-4">
              <Target className={result.model_type === 'regression' || !result.model_type ? "text-blue-500" : result.accuracy && result.accuracy >= 0.7 ? "text-green-500" : "text-amber-500"} size={32} />
            </div>
            
            {result.model_type === 'regression' ? (
                <>
                    <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wide">R² Score (Model Fit)</h3>
                    <p className="text-4xl font-extrabold text-slate-900 mt-2">
                    {result.r2_score?.toFixed(3) || "0.000"}
                    </p>
                    <span className="text-xs text-blue-600 font-medium mt-1 px-2 py-1 bg-blue-100 rounded-full">
                        Regression Model
                    </span>
                    <div className="mt-4 grid grid-cols-2 gap-2 text-xs w-full">
                        <div className="bg-white p-2 rounded border border-slate-200">
                            <div className="text-slate-400">RMSE</div>
                            <div className="font-bold">{result.rmse?.toFixed(2)}</div>
                        </div>
                        <div className="bg-white p-2 rounded border border-slate-200">
                            <div className="text-slate-400">MAE</div>
                            <div className="font-bold">{result.mae?.toFixed(2)}</div>
                        </div>
                        <div className="bg-white p-2 rounded border border-slate-200 col-span-2">
                            <div className="text-slate-400">MAPE</div>
                            <div className="font-bold">{(result.mape ? (result.mape * 100).toFixed(2) : "0.00")}%</div>
                        </div>
                    </div>
                </>
            ) : (
                <>
                    <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wide">Model Accuracy</h3>
                    <p className="text-4xl font-extrabold text-slate-900 mt-2">
                    {((result.accuracy || 0) * 100).toFixed(1)}%
                    </p>
                    {result.accuracy && result.accuracy >= 0.8 ? (
                    <span className="text-xs text-green-600 font-medium mt-1 px-2 py-1 bg-green-100 rounded-full">High Confidence</span>
                    ) : result.accuracy && result.accuracy >= 0.7 ? (
                    <span className="text-xs text-green-600 font-medium mt-1 px-2 py-1 bg-green-100 rounded-full">Good</span>
                    ) : (
                    <span className="text-xs text-amber-600 font-medium mt-1 px-2 py-1 bg-amber-100 rounded-full">Fair</span>
                    )}
                </>
            )}
            
            <div className="mt-3 text-xs text-slate-500">
               Test Set Analysis
            </div>
          </div>
          )}

          {/* Feature Importance Chart - Hide for Association Rules, PCA, FA, K-Means, Hierarchical */}
          {result.model_type !== 'association_rules' && result.model_type !== 'pca' && result.model_type !== 'factor-analysis' && result.model_type !== 'kmeans' && result.model_type !== 'hierarchical' && (
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
          )}

          {/* Confusion Matrix OR Regression Explanation */}
          {result.model_type !== 'regression' && result.model_type !== 'association_rules' && result.model_type !== 'pca' && result.model_type !== 'factor-analysis' && result.confusionMatrix && result.confusionMatrix.length > 0 && (
          <div className="lg:col-span-3 bg-white rounded-xl p-6 border border-slate-200">
            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Layers size={18} className="text-orange-500" />
              Confusion Matrix
            </h3>
            {/* ... Existing Confusion Matrix Code ... */}
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
              <div className="bg-indigo-50 p-4 rounded-lg text-center border border-indigo-100">
                <span className="block text-xs text-indigo-600 font-bold uppercase">True Positive</span>
                <span className="block text-2xl font-bold text-indigo-900">{result.confusionMatrix[0]?.[0] || 0}</span>
              </div>
              <div className="bg-red-50 p-4 rounded-lg text-center border border-red-100">
                <span className="block text-xs text-red-600 font-bold uppercase">False Positive</span>
                <span className="block text-2xl font-bold text-red-900">{result.confusionMatrix[0]?.[1] || 0}</span>
              </div>
              <div className="bg-red-50 p-4 rounded-lg text-center border border-red-100">
                <span className="block text-xs text-red-600 font-bold uppercase">False Negative</span>
                <span className="block text-2xl font-bold text-red-900">{result.confusionMatrix[1]?.[0] || 0}</span>
              </div>
              <div className="bg-indigo-50 p-4 rounded-lg text-center border border-indigo-100">
                <span className="block text-xs text-indigo-600 font-bold uppercase">True Negative</span>
                <span className="block text-2xl font-bold text-indigo-900">{result.confusionMatrix[1]?.[1] || 0}</span>
              </div>
            </div>
          </div>
          )}

          {/* Association Rules Table */}
          {result.model_type === 'association_rules' && result.associationRules && (
            <div className="lg:col-span-3 bg-white rounded-xl p-6 border border-slate-200 overflow-x-auto">
              <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                <List size={18} className="text-teal-500" />
                Top 50 Association Rules (Sorted by Lift)
              </h3>
              <table className="min-w-full text-sm text-left">
                <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                  <tr>
                    <th className="px-4 py-3">Antecedents (IF)</th>
                    <th className="px-4 py-3">Consequents (THEN)</th>
                    <th className="px-4 py-3 text-right">Support</th>
                    <th className="px-4 py-3 text-right">Confidence</th>
                    <th className="px-4 py-3 text-right">Lift</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {result.associationRules.map((rule, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="px-4 py-3 font-mono text-slate-700">{rule.antecedents}</td>
                      <td className="px-4 py-3 font-mono text-indigo-600">{rule.consequents}</td>
                      <td className="px-4 py-3 text-right text-slate-600">{rule.support.toFixed(3)}</td>
                      <td className="px-4 py-3 text-right text-slate-600">{rule.confidence.toFixed(3)}</td>
                      <td className="px-4 py-3 text-right font-bold text-emerald-600">{rule.lift.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* PCA Results Section */}
          {result.model_type === 'pca' && result.pcaResults && (
            <div className="lg:col-span-3 space-y-8">
              
              {/* 1. Explained Variance (Scree Plot) */}
              <div className="bg-white rounded-xl p-6 border border-slate-200">
                <h3 className="font-bold text-slate-900 mb-6 flex items-center gap-2">
                  <BarChart2 size={18} className="text-blue-600" />
                  Explained Variance by Component
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.pcaResults.explained_variance.map((val: number, idx: number) => ({ name: `PC${idx+1}`, value: val * 100 }))}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" tick={{fontSize: 12}} />
                      <YAxis label={{ value: '% Variance', angle: -90, position: 'insideLeft' }} />
                      <Tooltip contentStyle={{borderRadius: '8px'}} formatter={(val: number) => [`${val.toFixed(2)}%`, 'Variance']} />
                      <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                         {result.pcaResults.explained_variance.map((_: number, index: number) => (
                              <Cell key={`cell-${index}`} fill={['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'][index % 4] || '#bfdbfe'} />
                          ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-sm text-slate-500 mt-2 text-center">
                  Total Explained Variance: {(result.pcaResults.explained_variance.reduce((a: number, b: number) => a + b, 0) * 100).toFixed(1)}%
                </p>
              </div>

               {/* 2. Component Loadings (Table) */}
               <div className="bg-white rounded-xl p-6 border border-slate-200 overflow-x-auto">
                <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <List size={18} className="text-slate-600" />
                  Component Loadings (Key Drivers)
                </h3>
                <table className="min-w-full text-sm text-left">
                  <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                    <tr>
                      <th className="px-4 py-3">Component</th>
                      {result.features_used?.map((f: string) => (
                         <th key={f} className="px-4 py-3">{f}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                     {result.pcaResults.components.map((comp: any, idx: number) => (
                       <tr key={idx} className="hover:bg-slate-50">
                         <td className="px-4 py-3 font-bold text-slate-700">{comp.component}</td>
                         {result.features_used?.map((f: string) => (
                           <td key={f} className={`px-4 py-3 font-mono ${Math.abs(comp[f]) > 0.4 ? 'font-bold text-indigo-600' : 'text-slate-500'}`}>
                             {comp[f]?.toFixed(3)}
                           </td>
                         ))}
                       </tr>
                     ))}
                  </tbody>
                </table>
               </div>
            </div>
          )}

          {/* Factor Analysis Results Section */}
          {result.model_type === 'factor-analysis' && result.faResults && (
            <div className="lg:col-span-3 space-y-8">
              
              {/* 1. Factor Variance (Scree Plot Equivalent) */}
              <div className="bg-white rounded-xl p-6 border border-slate-200">
                <h3 className="font-bold text-slate-900 mb-6 flex items-center gap-2">
                  <BarChart2 size={18} className="text-teal-600" />
                  Factor Variance (Importance)
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.faResults.components.map(c => ({ name: c.component, value: c.variance }))}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" tick={{fontSize: 12}} />
                      <YAxis label={{ value: 'Variance (Sum sq loadings)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip contentStyle={{borderRadius: '8px'}} />
                      <Bar dataKey="value" fill="#0d9488" radius={[4, 4, 0, 0]}>
                         {result.faResults.components.map((_, index) => (
                              <Cell key={`cell-${index}`} fill={['#0d9488', '#14b8a6', '#5eead4', '#99f6e4'][index % 4] || '#99f6e4'} />
                          ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-sm text-slate-500 mt-2 text-center">
                  Log Likelihood: {result.faResults.log_likelihood.toFixed(2)}
                </p>
              </div>

               {/* 2. Factor Loadings (Table) */}
               <div className="bg-white rounded-xl p-6 border border-slate-200 overflow-x-auto">
                <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <List size={18} className="text-slate-600" />
                  Factor Loadings Matrix
                </h3>
                <table className="min-w-full text-sm text-left">
                  <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                    <tr>
                      <th className="px-4 py-3">Factor</th>
                      {result.features_used?.map(f => (
                         <th key={f} className="px-4 py-3">{f}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                     {result.faResults.components.map((comp, idx) => (
                       <tr key={idx} className="hover:bg-slate-50">
                         <td className="px-4 py-3 font-bold text-slate-700">{comp.component}</td>
                         {result.features_used?.map(f => (
                           <td key={f} className={`px-4 py-3 font-mono ${Math.abs(comp[f]) > 0.4 ? 'font-bold text-teal-600' : 'text-slate-500'}`}>
                             {comp[f]?.toFixed(3)}
                           </td>
                         ))}
                       </tr>
                     ))}
                  </tbody>
                </table>
               </div>

                {/* 3. Communalities */}
               <div className="bg-white rounded-xl p-6 border border-slate-200 overflow-x-auto">
                <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <Target size={18} className="text-purple-600" />
                  Communalities (Explained Variance per Feature)
                </h3>
                 <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(result.faResults.communalities).map(([feature, value]) => (
                    <div key={feature} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                      <div className="text-xs text-slate-500 mb-1">{feature}</div>
                      <div className="text-lg font-bold text-slate-800">{(value as number).toFixed(3)}</div>
                      <div className="w-full bg-slate-200 h-1.5 rounded-full mt-2">
                        <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${Math.min((value as number) * 100, 100)}%` }}></div>
                      </div>
                    </div>
                  ))}
                 </div>
               </div>

            </div>
          )}

          {/* K-Means Results Section */}
          {result.model_type === 'kmeans' && result.kmeansResults && (
            <div className="lg:col-span-3 space-y-6">
              
              {/* Summary Header */}
              <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-200">
                <h3 className="font-bold text-emerald-900 text-lg mb-4">🎯 K-Means 集群分析結果</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                  <div className="bg-white rounded-lg p-3 border border-emerald-100">
                    <div className="text-xs text-slate-500">集群數 (K)</div>
                    <div className="text-2xl font-bold text-emerald-700">{result.kmeansResults.n_clusters}</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-emerald-100">
                    <div className="text-xs text-slate-500">總數據點</div>
                    <div className="text-2xl font-bold text-slate-800">
                      {result.kmeansResults.cluster_distribution.reduce((sum: number, c: any) => sum + c.count, 0)}
                    </div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-emerald-100">
                    <div className="text-xs text-slate-500">特徵數</div>
                    <div className="text-2xl font-bold text-slate-800">{result.features_used?.length || 0}</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-emerald-100">
                    <div className="text-xs text-slate-500">WCSS (慣性)</div>
                    <div className="text-2xl font-bold text-amber-600">{result.kmeansResults.inertia.toFixed(2)}</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-emerald-100">
                    <div className="text-xs text-slate-500">輪廓係數</div>
                    <div className={`text-2xl font-bold ${(result.kmeansResults.silhouette_score || 0) > 0.5 ? 'text-green-600' : (result.kmeansResults.silhouette_score || 0) > 0.25 ? 'text-amber-600' : 'text-red-600'}`}>
                      {(result.kmeansResults.silhouette_score || 0).toFixed(3)}
                    </div>
                    <div className="text-xs text-slate-400">
                      {(result.kmeansResults.silhouette_score || 0) > 0.5 ? '優秀' : (result.kmeansResults.silhouette_score || 0) > 0.25 ? '尚可' : '較差'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Elbow Plot */}
              {result.kmeansResults.elbow_data && result.kmeansResults.elbow_data.length > 0 && (
              <div className="bg-white rounded-xl p-6 border border-slate-200">
                <h3 className="font-bold text-slate-900 mb-2">📈 手肘圖 (Elbow Plot)</h3>
                <p className="text-xs text-slate-500 mb-4">用於選擇最佳 K 值。尋找曲線彎曲點（手肘）作為最佳集群數。</p>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={result.kmeansResults.elbow_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="k" label={{ value: 'K (集群數)', position: 'insideBottomRight', offset: -5 }} />
                      <YAxis label={{ value: 'Inertia (WCSS)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip contentStyle={{borderRadius: '8px'}} />
                      <Line type="monotone" dataKey="inertia" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981', r: 5 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              )}
              
              {/* Cluster Sizes */}
              <div className="bg-white rounded-xl p-6 border border-slate-200">
                <h3 className="font-bold text-slate-900 mb-4">📊 各集群大小</h3>
                <div className="grid grid-cols-3 gap-4">
                  {result.kmeansResults.cluster_distribution.map((c: any) => {
                    const total = result.kmeansResults.cluster_distribution.reduce((sum: number, x: any) => sum + x.count, 0);
                    const percent = ((c.count / total) * 100).toFixed(1);
                    const colors = ['bg-emerald-100 text-emerald-800', 'bg-blue-100 text-blue-800', 'bg-purple-100 text-purple-800', 'bg-amber-100 text-amber-800'];
                    return (
                      <div key={c.cluster} className={`rounded-lg p-4 text-center ${colors[c.cluster % 4]}`}>
                        <div className="text-xs font-medium mb-1">集群 {c.cluster + 1}</div>
                        <div className="text-3xl font-bold">{c.count}</div>
                        <div className="text-xs mt-1">{percent}%</div>
                      </div>
                    );
                  })}
                </div>
              </div>

               {/* Cluster Centers (Centroids) */}
               <div className="bg-white rounded-xl p-6 border border-slate-200 overflow-x-auto">
                <h3 className="font-bold text-slate-900 mb-4">📍 集群中心點 (Centroids)</h3>
                <table className="min-w-full text-sm text-left">
                  <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                    <tr>
                      <th className="px-4 py-3">集群</th>
                      {result.features_used?.map((f: string) => (
                         <th key={f} className="px-4 py-3">{f}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                     {result.kmeansResults.cluster_centers.map((center: any, idx: number) => (
                       <tr key={idx} className="hover:bg-slate-50">
                         <td className="px-4 py-3 font-bold text-slate-700">集群 {center.cluster + 1}</td>
                         {result.features_used?.map((f: string) => (
                           <td key={f} className="px-4 py-3 font-mono text-slate-600">
                             {typeof center[f] === 'number' ? center[f].toLocaleString(undefined, {maximumFractionDigits: 2}) : center[f]}
                           </td>
                         ))}
                       </tr>
                     ))}
                  </tbody>
                </table>
               </div>
            </div>
          )}

          {/* Hierarchical Clustering Results Section */}
          {result.model_type === 'hierarchical' && result.hierarchicalResults && (
            <div className="lg:col-span-3 space-y-6">
              
              {/* Summary Header */}
              <div className="bg-purple-50 rounded-xl p-6 border border-purple-200">
                <h3 className="font-bold text-purple-900 text-lg mb-4">🌳 階層式集群分析結果</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div className="bg-white rounded-lg p-3 border border-purple-100">
                    <div className="text-xs text-slate-500">集群數 (K)</div>
                    <div className="text-2xl font-bold text-purple-700">{result.hierarchicalResults.n_clusters}</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-purple-100">
                    <div className="text-xs text-slate-500">總數據點</div>
                    <div className="text-2xl font-bold text-slate-800">
                      {result.hierarchicalResults.cluster_distribution.reduce((sum: number, c: any) => sum + c.count, 0)}
                    </div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-purple-100">
                    <div className="text-xs text-slate-500">特徵數</div>
                    <div className="text-2xl font-bold text-slate-800">{result.features_used?.length || 0}</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-purple-100">
                    <div className="text-xs text-slate-500">連結方法</div>
                    <div className="text-xl font-bold text-purple-600">{result.hierarchicalResults.linkage}</div>
                  </div>
                </div>
              </div>

              {/* Educational Note */}
              <div className="bg-purple-100/50 rounded-xl p-4 border border-purple-200">
                <h4 className="font-semibold text-purple-800 text-sm mb-2">💡 什麼是階層式集群？</h4>
                <p className="text-xs text-purple-700 leading-relaxed">
                  階層式集群是一種「由下而上」的聚合方法。它從每個數據點作為獨立集群開始，逐步合併最相似的集群，直到達到指定的集群數量。
                  <strong className="block mt-2">Ward 連結法</strong>會最小化合併時的方差增量，產生較為緊湊、大小相近的集群。
                </p>
              </div>
              
              {/* Cluster Sizes */}
              <div className="bg-white rounded-xl p-6 border border-slate-200">
                <h3 className="font-bold text-slate-900 mb-4">📊 各集群大小</h3>
                <div className="grid grid-cols-3 gap-4">
                  {result.hierarchicalResults.cluster_distribution.map((c: any) => {
                    const total = result.hierarchicalResults!.cluster_distribution.reduce((sum: number, x: any) => sum + x.count, 0);
                    const percent = ((c.count / total) * 100).toFixed(1);
                    const colors = ['bg-purple-100 text-purple-800', 'bg-indigo-100 text-indigo-800', 'bg-violet-100 text-violet-800', 'bg-fuchsia-100 text-fuchsia-800'];
                    return (
                      <div key={c.cluster} className={`rounded-lg p-4 text-center ${colors[c.cluster % 4]}`}>
                        <div className="text-xs font-medium mb-1">集群 {c.cluster + 1}</div>
                        <div className="text-3xl font-bold">{c.count}</div>
                        <div className="text-xs mt-1">{percent}%</div>
                      </div>
                    );
                  })}
                </div>
              </div>

               {/* Cluster Centers */}
               <div className="bg-white rounded-xl p-6 border border-slate-200 overflow-x-auto">
                <h3 className="font-bold text-slate-900 mb-4">📍 集群中心點 (Centroids)</h3>
                <table className="min-w-full text-sm text-left">
                  <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
                    <tr>
                      <th className="px-4 py-3">集群</th>
                      {result.features_used?.map((f: string) => (
                         <th key={f} className="px-4 py-3">{f}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                     {result.hierarchicalResults.cluster_centers.map((center: any, idx: number) => (
                       <tr key={idx} className="hover:bg-slate-50">
                         <td className="px-4 py-3 font-bold text-slate-700">集群 {center.cluster + 1}</td>
                         {result.features_used?.map((f: string) => (
                           <td key={f} className="px-4 py-3 font-mono text-slate-600">
                             {typeof center[f] === 'number' ? center[f].toLocaleString(undefined, {maximumFractionDigits: 2}) : center[f]}
                           </td>
                         ))}
                       </tr>
                     ))}
                  </tbody>
                </table>
               </div>
            </div>
          )}

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