import React, { useState } from 'react';
import { Brain, AlertCircle, BarChart2, Clock } from 'lucide-react';
import { Dataset, AnalysisStatus, ModelResult, StatisticsData } from './types';
import DataIngestion from './components/DataIngestion';
import DataPreview from './components/DataPreview';
import EDAStats from './components/EDAStats';
import DataVisualization from './components/DataVisualization';
import FeatureSelection from './components/FeatureSelection';
import ResultsDashboard from './components/ResultsDashboard';
import PredictionPanel from './components/PredictionPanel';
import AnalysisTabs, { AnalysisTab } from './components/AnalysisTabs';
import MachineLearningMenu, { MLModel } from './components/MachineLearningMenu';
import BasicAnalysisMenu, { BasicAnalysisType } from './components/BasicAnalysisMenu';
import TimeSeriesMenu, { TimeSeriesType } from './components/TimeSeriesMenu';
import ParetoAnalysis from './components/ParetoAnalysis';
import HypothesisTesting from './components/HypothesisTesting';
import ChiSquareTest from './components/ChiSquareTest';
import RegressionAnalysis from './components/RegressionAnalysis';
import CorrelationMatrix from './components/CorrelationMatrix';

// Environment-aware API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
console.log('🔗 Using API URL:', API_URL); // Debug: see which URL is being used

const App: React.FC = () => {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [status, setStatus] = useState<AnalysisStatus>(AnalysisStatus.IDLE);
  const [result, setResult] = useState<ModelResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [statistics, setStatistics] = useState<StatisticsData | null>(null);
  const [statsLoading, setStatsLoading] = useState<boolean>(false);
  const [statsError, setStatsError] = useState<string>('');

  // Navigation State
  const [activeTab, setActiveTab] = useState<AnalysisTab>('data-exploration');
  const [activeMLModel, setActiveMLModel] = useState<MLModel>('random-forest');
  const [activeBasicAnalysis, setActiveBasicAnalysis] = useState<BasicAnalysisType>('correlation-matrix');
  const [activeTimeSeriesModel, setActiveTimeSeriesModel] = useState<TimeSeriesType>('holt-winters');

  const handleDataLoaded = async (newDataset: Dataset) => {
    setDataset(newDataset);
    setResult(null);
    setStatus(AnalysisStatus.IDLE);
    setErrorMessage('');
    setStatsError(''); // Reset stats error
    
    // Fetch statistics
    setStatsLoading(true);
    try {
      console.log('Sending statistics request to:', `${API_URL}/statistics`);
      const response = await fetch(`${API_URL}/statistics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: newDataset.rows })
      });
      
      if (response.ok) {
        const statsData = await response.json();
        setStatistics(statsData);
      } else {
        const errorText = await response.text();
        console.error('Statistics fetch failed:', response.status, errorText);
        let errorMsg = `Server error: ${response.status}`;
        try {
            const jsonError = JSON.parse(errorText);
            if (jsonError && jsonError.detail) {
                // Handle Pydantic validation errors (array of objects)
                if (typeof jsonError.detail === 'object') {
                    errorMsg = JSON.stringify(jsonError.detail);
                } else {
                    errorMsg = jsonError.detail;
                }
            }
        } catch (e) { /* ignore json parse error */ }
        setStatsError(errorMsg);
      }
    } catch (e: any) {
      console.error('Failed to fetch statistics:', e);
      setStatsError(e.message || "Failed to connect to backend for statistics");
    } finally {
      setStatsLoading(false);
    }
  };

  // The function to call the Python backend
  const runAnalysis = async (target: string, features: string[]) => {
    setStatus(AnalysisStatus.LOADING);
    try {
      
      let endpoint = '/train-rf';
      if (activeMLModel === 'logistic-regression') {
        endpoint = '/train-lr';
      } else if (activeMLModel === 'decision-tree') {
        endpoint = '/train-dt';
      } else if (activeMLModel === 'neural-network') {
        endpoint = '/train-nn';
      } else if (activeMLModel === 'apriori') {
        endpoint = '/train-apriori';
      } else if (activeMLModel === 'pca') {
        endpoint = '/train-pca';
      } else if (activeMLModel === 'factor-analysis') {
        endpoint = '/train-fa';
      } else if (activeMLModel === 'kmeans') {
        endpoint = '/train-kmeans';
      } else if (activeMLModel === 'hierarchical') {
        endpoint = '/train-hierarchical';
      }

      // Use the environment-aware URL
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset!.rows,
          target,
          features
        })
      });

      if (!response.ok) {
        // Try to parse the JSON error details from FastAPI
        let errorDetail = response.statusText;
        try {
          const errorJson = await response.json();
          if (errorJson && errorJson.detail) {
            errorDetail = errorJson.detail;
          }
        } catch (e) {
          // If response isn't JSON, just use status text
        }
        throw new Error(errorDetail);
      }

      const data = await response.json();
      
      setResult({
        model_type: data.model_type,
        accuracy: data.metrics.accuracy,
        precision: data.metrics.precision,
        recall: data.metrics.recall,
        f1_score: data.metrics.f1_score,
        // Regression metrics (optional)
        r2_score: data.metrics.r2_score,
        rmse: data.metrics.rmse,
        mae: data.metrics.mae,
        mape: data.metrics.mape,
        
        featureImportance: data.feature_importance,
        confusionMatrix: data.confusion_matrix,
        warning: data.warning || null,
        features_used: data.features_used || [],
        features_skipped: data.features_skipped || [],
        predictions: [],
        associationRules: data.association_rules,
        pcaResults: data.pca_results,
        faResults: data.fa_results,
        kmeansResults: data.kmeans_results,
        hierarchicalResults: data.hierarchical_results
      });
      setStatus(AnalysisStatus.SUCCESS);
    } catch (e: any) {
      console.error(e);
      setErrorMessage(e.message || "Failed to connect to Python backend.");
      setStatus(AnalysisStatus.ERROR);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans">
      {/* Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="bg-indigo-600 text-white p-2 rounded-lg">
                <Brain size={24} />
              </div>
              <span className="font-bold text-xl tracking-tight text-slate-900">Python Data Analysis</span>
            </div>
            <div className="flex items-center gap-4 text-sm font-medium text-slate-500">
              <span className="hidden sm:block">Python + React Architecture</span>
              <a href="#" className="text-indigo-600 hover:text-indigo-800">Documentation</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero / Header Section - Slightly reduced bottom margin to fit tabs */}
      <div className="bg-white border-b border-slate-100 pb-8 pt-10">
         <div className="text-center max-w-3xl mx-auto px-4">
          <h1 className="text-4xl font-extrabold text-slate-900 sm:text-5xl mb-4">
            Data Analysis & ML Studio
          </h1>
          <p className="text-lg text-slate-600">
            Professional tool for data ingestion and Random Forest modeling.
          </p>
        </div>
      </div>

      <main className="min-h-[calc(100vh-200px)]">
        
        {/* 1. Data Ingestion (Always Visible) */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="animate-fade-in-up">
            <DataIngestion onDataLoaded={handleDataLoaded} />
          </div>
        </div>

        {/* 1.5. Data Preview (Always Visible if dataset exists) */}
        {dataset && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-8">
            <div className="animate-fade-in-up delay-75">
              <DataPreview dataset={dataset} />
            </div>
          </div>
        )}

        {/* TABS NAVIGATION */}
        {dataset && (
          <>
            <AnalysisTabs 
              activeTab={activeTab} 
              onTabChange={setActiveTab} 
            />
            
            {/* Sub-menu for Machine Learning */}
            {activeTab === 'machine-learning' && (
              <MachineLearningMenu
                activeModel={activeMLModel}
                onModelChange={setActiveMLModel}
              />
            )}
          </>
        )}

        {/* MAIN CONTENT AREA */}
        {dataset && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
            
            {/* CONTENT: DATA EXPLORATION */}
            {activeTab === 'data-exploration' && (
              <>
                <div className="animate-fade-in-up delay-100">
                  {statsError ? (
                      <div className="bg-red-50 p-4 rounded-lg border border-red-200 mb-6">
                          <div className="flex items-center gap-2 text-red-700 font-bold mb-2">
                              <AlertCircle size={20} />
                              Statistics Failed to Load
                          </div>
                          <p className="text-sm text-red-600 font-mono">{statsError}</p>
                      </div>
                  ) : (
                      <EDAStats statistics={statistics} isLoading={statsLoading} />
                  )}
                </div>

                <div className="animate-fade-in-up delay-150">
                  <DataVisualization statistics={statistics} isLoading={statsLoading} dataset={dataset} />
                </div>
              </>
            )}

            {/* CONTENT: BASIC ANALYSIS */}
            {activeTab === 'basic-analysis' && (
              <>
                <BasicAnalysisMenu 
                  activeAnalysis={activeBasicAnalysis} 
                  onAnalysisChange={setActiveBasicAnalysis} 
                />
                
                {/* Pareto Analysis */}
                {activeBasicAnalysis === 'pareto-analysis' && dataset ? (
                  <ParetoAnalysis dataset={dataset} apiUrl={API_URL} />
                ) : activeBasicAnalysis === 'hypothesis-testing' && dataset ? (
                  <HypothesisTesting dataset={dataset} apiUrl={API_URL} />
                ) : activeBasicAnalysis === 'chi-square' && dataset ? (
                  <ChiSquareTest dataset={dataset} apiUrl={API_URL} />
                ) : activeBasicAnalysis === 'regression' && dataset ? (
                  <RegressionAnalysis dataset={dataset} apiUrl={API_URL} />
                ) : activeBasicAnalysis === 'correlation-matrix' && dataset ? (
                  <CorrelationMatrix dataset={dataset} apiUrl={API_URL} />
                ) : (
                  <div className="text-center py-20 bg-white rounded-xl border border-dashed border-slate-300 mt-4">
                    <BarChart2 size={48} className="mx-auto text-slate-300 mb-4" />
                    <h3 className="text-lg font-medium text-slate-900">{activeBasicAnalysis} - Coming Soon</h3>
                    <p className="text-slate-500">This analysis module is under development.</p>
                  </div>
                )}
              </>
            )}

             {/* CONTENT: TIME SERIES */}
             {activeTab === 'time-series' && (
              <>
                <TimeSeriesMenu 
                  activeModel={activeTimeSeriesModel} 
                  onModelChange={setActiveTimeSeriesModel} 
                />
                <div className="text-center py-20 bg-white rounded-xl border border-dashed border-slate-300 mt-4">
                  <Clock size={48} className="mx-auto text-slate-300 mb-4" />
                  <h3 className="text-lg font-medium text-slate-900">{activeTimeSeriesModel} - Coming Soon</h3>
                  <p className="text-slate-500">This time series model is under development.</p>
                </div>
              </>
            )}

            {/* CONTENT: MACHINE LEARNING */}
            {activeTab === 'machine-learning' && (
              <>
                {/* Random Forest OR Logistic Regression OR Decision Tree OR Neural Network OR Apriori OR PCA OR Factor Analysis OR K-Means OR Hierarchical */}
                {activeMLModel === 'random-forest' || activeMLModel === 'logistic-regression' || activeMLModel === 'decision-tree' || activeMLModel === 'neural-network' || activeMLModel === 'apriori' || activeMLModel === 'pca' || activeMLModel === 'factor-analysis' || activeMLModel === 'kmeans' || activeMLModel === 'hierarchical' ? (
                  <div className="animate-fade-in-up">
                      <FeatureSelection 
                        headers={dataset.headers} 
                        onRunAnalysis={runAnalysis}
                        isProcessing={status === AnalysisStatus.LOADING}
                        activeModel={activeMLModel}
                        buttonText={
                        activeMLModel === 'logistic-regression' 
                          ? 'Train Logistic Regression' 
                          : activeMLModel === 'decision-tree'
                            ? 'Train Decision Tree'
                            : activeMLModel === 'neural-network'
                              ? 'Train Neural Network'
                              : activeMLModel === 'apriori'
                                ? 'Train Apriori'
                                : activeMLModel === 'pca'
                                  ? 'Run PCA Analysis'
                                  : activeMLModel === 'factor-analysis'
                                    ? 'Run Factor Analysis' 
                                    : activeMLModel === 'kmeans'
                                      ? 'Run K-Means Clustering'
                                      : activeMLModel === 'hierarchical'
                                        ? 'Run Hierarchical Clustering'
                                        : 'Train Random Forest'
                      }
                    />

                    {/* Error Message */}
                    {status === AnalysisStatus.ERROR && (
                      <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-md animate-shake shadow-sm mt-6">
                        <div className="flex items-start">
                          <AlertCircle className="text-red-500 mt-0.5" size={20} />
                          <div className="ml-3">
                            <h3 className="text-sm font-bold text-red-800">Analysis Failed</h3>
                            <p className="text-sm text-red-700 mt-1 font-mono bg-red-100/50 p-2 rounded">
                              {errorMessage}
                            </p>
                            <p className="text-xs text-red-500 mt-2">
                              Check your terminal running 'server.py' for detailed logs.
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Results Section */}
                    {result && status === AnalysisStatus.SUCCESS && (
                      <div className="animate-fade-in-up delay-300 mt-8 space-y-8">
                        <ResultsDashboard result={result} />
                        {/* 4. Model Prediction (Hide for Apriori, PCA, FA, K-Means, Hierarchical) */}
                        {activeMLModel !== 'apriori' && activeMLModel !== 'pca' && activeMLModel !== 'factor-analysis' && activeMLModel !== 'kmeans' && activeMLModel !== 'hierarchical' && (
                        <PredictionPanel 
                          features={result.features_used || []} 
                          featuresUsed={result.features_used || []} 
                          statistics={statistics}
                        />
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  /* Placeholder for other ML models */
                  <div className="text-center py-20 bg-white rounded-xl border border-dashed border-slate-300">
                    <Brain size={48} className="mx-auto text-slate-300 mb-4" />
                    <h3 className="text-lg font-medium text-slate-900">Model Coming Soon</h3>
                    <p className="text-slate-500">The selected model {activeMLModel} is not yet implemented.</p>
                  </div>
                )}
              </>
            )}

          </div>
        )}

        {/* Loading State Overlay (Global) */}
        {status === AnalysisStatus.LOADING && !result && activeTab === 'machine-learning' && (
          <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto mb-4"></div>
              <h3 className="text-xl font-bold text-indigo-900">Running Python Kernel...</h3>
              <p className="text-slate-500">
                {activeMLModel === 'logistic-regression' 
                   ? 'Training Logistic Regression Model...' 
                   : activeMLModel === 'decision-tree'
                     ? 'Training Decision Tree Model...'
                     : activeMLModel === 'neural-network'
                       ? 'Training Neural Network...'
                       : activeMLModel === 'apriori'
                         ? 'Mining Association Rules...'
                         : activeMLModel === 'pca'
                           ? 'Running PCA Analysis...'
                           : activeMLModel === 'factor-analysis'
                             ? 'Running Factor Analysis...'
                             : activeMLModel === 'kmeans'
                               ? 'Running K-Means Clustering...'
                               : activeMLModel === 'hierarchical'
                                 ? 'Running Hierarchical Clustering...'
                                 : 'Training Random Forest (n_estimators=100)'}
              </p>
            </div>
          </div>
        )}

      </main>
    </div>
  );
};

export default App;