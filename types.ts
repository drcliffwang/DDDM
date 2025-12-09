export interface DataRow {
  [key: string]: string | number | boolean | null;
}

export interface Dataset {
  fileName: string;
  headers: string[];
  rows: DataRow[];
}

export interface ModelResult {
  model_type?: 'classification' | 'regression';
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  // Regression metrics
  r2_score?: number;
  rmse?: number;
  mae?: number;
  mape?: number;
  
  featureImportance: Array<{ name: string; value: number }>;
  confusionMatrix: number[][]; // Empty for regression
  warning?: string | null;
  features_used?: string[];
  features_skipped?: string[];
  predictions: number[];
}

export enum AnalysisStatus {
  IDLE = 'IDLE',
  LOADING = 'LOADING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

export interface StatisticsData {
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
  missing_values: {
    [key: string]: {
      count: number;
      percentage: number;
    };
  };
  correlation_matrix: {
    columns: string[];
    values: number[][];
  };
}

