from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

app = FastAPI()

# Global storage for trained model
model_artifact = {}

# Get environment (Railway sets this automatically)
environment = os.getenv('RAILWAY_ENVIRONMENT', 'development')

# CORS - allow both local and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",           # Local React dev server
        "https://dddm-smoky.vercel.app",   # Your Vercel deployment
        "https://*.vercel.app",            # Any Vercel deployment (fallback)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug: See which environment we're in
@app.get("/")
def read_root():
    return {
        "app": "DDDM Backend",
        "environment": environment,
        "status": "running"
    }

class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    target: str
    features: List[str]

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# ... (imports)

@app.post("/train-rf")
def train_random_forest(request: TrainRequest):
    try:
        # 1. Load Data
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")

        # 2. Check Columns
        if request.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target}' not found.")
        
        missing_features = [f for f in request.features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

        # 3. Data Cleaning
        df = df.dropna(subset=[request.target])
        if df.empty:
            raise HTTPException(status_code=400, detail="All rows have missing target values.")

        available_columns = set(df.columns)
        requested_features = set(request.features)
        requested_target = request.target
        
        missing_features = requested_features - available_columns
        available_features = list(requested_features & available_columns)
        
        if len(available_features) == 0:
            raise HTTPException(status_code=400, detail="No available features found.")
        
        warning_message = None
        if missing_features:
            warning_message = f"Warning: {len(missing_features)} feature(s) skipped."

        X = df[available_features]
        y = df[requested_target]

        # Handle X (Features)
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='ignore')
                except:
                    pass
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        X = pd.get_dummies(X)

        # 4. Determine Problem Type (Classification vs Regression)
        # Heuristic: If target is numeric and has many unique values -> Regression
        # If target is object/string OR numeric with few unique values -> Classification
        is_regression = False
        
        if pd.api.types.is_numeric_dtype(y):
            unique_count = len(y.unique())
            # If float values, assume regression
            if pd.api.types.is_float_dtype(y):
                is_regression = True
            # If integer but many unique values (e.g. > 20), assume regression
            elif unique_count > 20:
                is_regression = True
            else:
                is_regression = False # Treat low cardinality int as classification
        else:
            is_regression = False # Object/String is classification

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        metrics = {}
        cm_list = []
        
        if is_regression:
            # REGRESSION
            # Ensure y is numeric
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(0)
            
            clf = RandomForestRegressor(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            metrics = {
                "r2_score": float(r2),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape)
            }
            
        else:
            # CLASSIFICATION
            y = y.astype(str) # Force string for classification
            # Re-split to ensure y consistency
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if len(y.unique()) < 2:
                raise HTTPException(status_code=400, detail="Target variable needs at least 2 different classes.")

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
            cm_list = cm.tolist()

        # Save model artifact
        global model_artifact
        model_artifact = {
            "model": clf,
            "train_columns": X.columns.tolist(),
            "target": requested_target,
            "type": "regression" if is_regression else "classification"
        }

        importances = clf.feature_importances_
        feature_imp_list = [
            {"name": str(name), "value": float(imp)} 
            for name, imp in zip(X.columns, importances)
        ]
        feature_imp_list.sort(key=lambda x: x['value'], reverse=True)

        return {
            "status": "success",
            "model_type": "regression" if is_regression else "classification",
            "metrics": metrics,
            "feature_importance": feature_imp_list,
            "confusion_matrix": cm_list,
            "warning": warning_message,
            "features_used": available_features,
            "features_skipped": list(missing_features) if missing_features else []
        }

    except Exception as e:
        # This print statement will show up in your Terminal where uvicorn is running
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/predict")
def predict_model(request: dict):
    """Predict target using trained model"""
    global model_artifact
    
    if not model_artifact:
        raise HTTPException(status_code=400, detail="No model trained yet. Please train a model first.")
    
    try:
        data = request.get('data')
        if not data:
            raise HTTPException(status_code=400, detail="No data provided for prediction.")

        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Handle numeric conversion (same as training)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # One-Hot Encoding
        df = pd.get_dummies(df)
        
        # Align columns with training data
        train_cols = model_artifact['train_columns']
        # Add missing columns with 0
        for col in train_cols:
            if col not in df.columns:
                df[col] = 0
                
        # select only training columns in correct order
        X = df[train_cols]
        X = X.fillna(0)
        
        # Predict
        clf = model_artifact['model']
        prediction = clf.predict(X)[0]
        
        # Get probability if available
        try:
            proba = clf.predict_proba(X)[0].tolist()
            classes = clf.classes_.tolist()
            probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
        except:
            probabilities = None
            
        return {
            "status": "success",
            "prediction": str(prediction),
            "probabilities": probabilities
        }

    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

class StatisticsRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/statistics")
def get_statistics(request: StatisticsRequest):
    """Calculate descriptive statistics for the dataset"""
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Descriptive statistics for numeric columns
        numeric_stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe()
            for col in numeric_cols:
                numeric_stats[col] = {
                    "count": int(desc[col]['count']) if not pd.isna(desc[col]['count']) else 0,
                    "mean": float(desc[col]['mean']) if not pd.isna(desc[col]['mean']) else 0,
                    "std": float(desc[col]['std']) if not pd.isna(desc[col]['std']) else 0,
                    "min": float(desc[col]['min']) if not pd.isna(desc[col]['min']) else 0,
                    "q1": float(desc[col]['25%']) if not pd.isna(desc[col]['25%']) else 0,
                    "median": float(desc[col]['50%']) if not pd.isna(desc[col]['50%']) else 0,
                    "q3": float(desc[col]['75%']) if not pd.isna(desc[col]['75%']) else 0,
                    "max": float(desc[col]['max']) if not pd.isna(desc[col]['max']) else 0,
                }
        
        # Missing values
        missing_values = {}
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_pct = float((missing_count / len(df)) * 100)
            missing_values[col] = {
                "count": missing_count,
                "percentage": round(missing_pct, 2)
            }
        
        # Correlation matrix for numeric columns
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            correlation_matrix = {
                "columns": numeric_cols,
                "values": corr.fillna(0).values.tolist()
            }
        
        # Value counts for ALL columns (top 10 common values for suggestions)
        categorical_stats = {}
        for col in df.columns:
            value_counts = df[col].value_counts().head(10)
            categorical_stats[col] = [
                {"value": str(val), "count": int(count)}
                for val, count in value_counts.items()
            ]
        
        return {
            "status": "success",
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols)
            },
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "missing_values": missing_values,
            "correlation_matrix": correlation_matrix
        }
    
    except Exception as e:
        print(f"STATISTICS ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Backend is running"}