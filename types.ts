export interface DataRow {
  [key: string]: string | number | boolean | null;
}

export interface Dataset {
  fileName: string;
  headers: string[];
  rows: DataRow[];
}

export interface ModelResult {
  model_type: 'classification' | 'regression' | 'association_rules' | 'pca' | 'factor-analysis' | 'kmeans' | 'hierarchical';
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
  associationRules?: AssociationRule[];
  pcaResults?: {
    explained_variance: number[];
    components: any[];
    projected_data: any[];
  };
  faResults?: {
    components: any[];
    communalities: {[key: string]: number};
    noise_variance: number[];
    log_likelihood: number;
  };
  kmeansResults?: {
    cluster_distribution: Array<{cluster: number; count: number}>;
    cluster_centers: any[];
    inertia: number;
    n_clusters: number;
    silhouette_score?: number;
    elbow_data?: Array<{k: number; inertia: number}>;
  };
  hierarchicalResults?: {
    cluster_distribution: Array<{cluster: number; count: number}>;
    cluster_centers: any[];
    n_clusters: number;
    linkage: string;
  };
}

export type AssociationRule = {
    antecedents: string;
    consequents: string;
    support: number;
    confidence: number;
    lift: number;
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

