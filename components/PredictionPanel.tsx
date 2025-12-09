import React, { useState } from 'react';
import { Send, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { StatisticsData } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Props {
  features: string[];
  featuresUsed: string[];
  statistics: StatisticsData | null;
  onClose?: () => void;
}

interface PredictionResult {
  prediction: string;
  probabilities?: Record<string, number>;
}

const PredictionPanel: React.FC<Props> = ({ featuresUsed, statistics }) => {
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (feature: string, value: string) => {
    setInputs(prev => ({
      ...prev,
      [feature]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Filter inputs to only include used features
      const predictionData: Record<string, any> = {};
      featuresUsed.forEach(feature => {
        const val = inputs[feature];
        // Try convert to number if it looks like one
        if (val && !isNaN(Number(val))) {
          predictionData[feature] = Number(val);
        } else {
          predictionData[feature] = val || "";
        }
      });

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: predictionData }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const data = await response.json();
      setResult({
        prediction: data.prediction,
        probabilities: data.probabilities
      });

    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-purple-500 mt-8">
      <div className="p-8">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 mb-6">
          <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-sm font-bold text-slate-600">4</div>
          Model Prediction (模型預測)
        </h2>
        
        <p className="text-slate-500 mb-6 ml-10">
          Enter values for features to predict the target using the trained model.
        </p>

        <div className="ml-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {featuresUsed.map(feature => {
            const hasSuggestions = statistics?.categorical_stats?.[feature];
            return (
              <div key={feature} className="space-y-2">
                <label className="text-sm font-medium text-slate-700 block truncate" title={feature}>
                  {feature}
                </label>
                <div className="relative">
                  <input
                    type="text"
                    list={hasSuggestions ? `list-${feature}` : undefined}
                    placeholder={`Value for ${feature}`}
                    value={inputs[feature] || ''}
                    onChange={(e) => handleInputChange(feature, e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                    autoComplete="off"
                  />
                  {hasSuggestions && (
                    <datalist id={`list-${feature}`}>
                      {hasSuggestions.map((item) => (
                        <option key={item.value} value={item.value} />
                      ))}
                    </datalist>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="ml-10">
          <button
            onClick={handlePredict}
            disabled={loading}
            className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 className="animate-spin" size={18} /> : <Send size={18} />}
            Predict Result
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          {result && (
            <div className="mt-6 p-6 bg-purple-50 rounded-xl border border-purple-100 animate-fade-in">
              <div className="flex items-center gap-3 mb-2">
                <CheckCircle className="text-purple-600" size={24} />
                <h3 className="text-lg font-bold text-purple-900">Prediction Result</h3>
              </div>
              
              <div className="ml-9">
                <div className="text-3xl font-extrabold text-purple-700 mb-2">
                  {result.prediction}
                </div>
                
                {result.probabilities && (
                  <div className="mt-4 space-y-2">
                    <p className="text-sm font-medium text-purple-800 opacity-80 mb-2">Class Probabilities:</p>
                    {Object.entries(result.probabilities).map(([cls, prob]) => (
                      <div key={cls} className="flex items-center gap-3">
                        <span className="w-16 text-sm font-medium text-slate-600 truncate">{cls}</span>
                        <div className="flex-1 h-2 bg-purple-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-purple-600 rounded-full" 
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold text-purple-700">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
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

export default PredictionPanel;
