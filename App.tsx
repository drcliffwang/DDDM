import React, { useState } from 'react';
import { Brain, AlertCircle } from 'lucide-react';
import { Dataset, AnalysisStatus, ModelResult, StatisticsData } from './types';
import DataIngestion from './components/DataIngestion';
import DataPreview from './components/DataPreview';
import EDAStats from './components/EDAStats';
import DataVisualization from './components/DataVisualization';
import FeatureSelection from './components/FeatureSelection';
import ResultsDashboard from './components/ResultsDashboard';

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

  const handleDataLoaded = async (newDataset: Dataset) => {
    setDataset(newDataset);
    // Fetch statistics
    setStatsLoading(true);
    try {
      // Use the environment-aware URL
      const response = await fetch(`${API_URL}/statistics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: newDataset.rows })
      });
      
      if (response.ok) {
        const statsData = await response.json();
        setStatistics(statsData);
      }
    } catch (e) {
      console.error('Failed to fetch statistics:', e);
    } finally {
      setStatsLoading(false);
    }
  };

  // The function to call the Python backend
  const runAnalysis = async (target: string, features: string[]) => {
    setStatus(AnalysisStatus.LOADING);
    try {
      // Use the environment-aware URL
      const response = await fetch(`${API_URL}/train-rf`, {
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
        accuracy: data.metrics.accuracy,
        precision: data.metrics.precision,
        recall: data.metrics.recall,
        f1_score: data.metrics.f1_score,
        featureImportance: data.feature_importance,
        confusionMatrix: data.confusion_matrix,
        warning: data.warning || null,
        features_used: data.features_used || [],
        features_skipped: data.features_skipped || [],
        predictions: [] 
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

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
        
        {/* Header Section */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h1 className="text-4xl font-extrabold text-slate-900 sm:text-5xl mb-4">
            Data Analysis & ML Studio
          </h1>
          <p className="text-lg text-slate-600">
            Professional tool for data ingestion and Random Forest modeling.
          </p>
        </div>

        {/* 1. Data Ingestion (New UI) */}
        <div className="animate-fade-in-up">
          <DataIngestion onDataLoaded={handleDataLoaded} />
        </div>

        {/* 1.5. Data Preview */}
        {dataset && (
          <div className="animate-fade-in-up delay-75">
            <DataPreview dataset={dataset} />
          </div>
        )}

        {/* 1.6. EDA Statistics */}
        {dataset && (
          <div className="animate-fade-in-up delay-100">
            <EDAStats statistics={statistics} isLoading={statsLoading} />
          </div>
        )}

        {/* 1.7. Data Visualization */}
        {dataset && (
          <div className="animate-fade-in-up delay-150">
            <DataVisualization statistics={statistics} isLoading={statsLoading} dataset={dataset} />
          </div>
        )}

        {/* 2. Configuration Section */}
        {dataset && (
          <div className="animate-fade-in-up delay-200">
             <FeatureSelection 
               headers={dataset.headers} 
               onRunAnalysis={runAnalysis}
               isProcessing={status === AnalysisStatus.LOADING}
             />
          </div>
        )}

        {/* Error Message */}
        {status === AnalysisStatus.ERROR && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-md animate-shake shadow-sm">
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

        {/* 3. Results Section */}
        {result && status === AnalysisStatus.SUCCESS && (
          <div className="animate-fade-in-up delay-300">
            <ResultsDashboard result={result} />
          </div>
        )}

        {/* Loading State Overlay */}
        {status === AnalysisStatus.LOADING && !result && (
          <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto mb-4"></div>
              <h3 className="text-xl font-bold text-indigo-900">Running Python Kernel...</h3>
              <p className="text-slate-500">Training Random Forest (n_estimators=100)</p>
            </div>
          </div>
        )}

      </main>
    </div>
  );
};

export default App;