from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

app = FastAPI()

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

        # 3. Data Cleaning (Aggressive)
        # Drop rows where Target is missing
        df = df.dropna(subset=[request.target])
        
        if df.empty:
            raise HTTPException(status_code=400, detail="All rows have missing target values.")
        # 2. Validate and filter features
        available_columns = set(df.columns)
        requested_features = set(request.features)
        requested_target = request.target
        
        # Check which features are missing
        missing_features = requested_features - available_columns
        available_features = list(requested_features & available_columns)
        
        # Check if target exists
        if requested_target not in available_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{requested_target}' not found in dataset. Available columns: {list(df.columns)}"
            )
        
        # If no features are available, fail
        if len(available_features) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"None of the requested features exist in the dataset. Requested: {list(requested_features)}, Available: {list(available_columns)}"
            )
        
        # Warn about missing features but continue with available ones
        warning_message = None
        if missing_features:
            warning_message = f"Warning: {len(missing_features)} feature(s) not found and were skipped: {list(missing_features)}"
            print(f"⚠️ {warning_message}")
        
        # 3. Extract features and target (only using available features)
        X = df[available_features]
        y = df[requested_target]

        # Handle numeric columns that might have strings (force coerce)
        for col in X.columns:
            # If column is object type, try to convert to numeric, coerce errors to NaN
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='ignore')
                except:
                    pass

        # Fill NaNs with 0
        X = X.fillna(0)
        
        # Replace Infinity with 0
        X = X.replace([np.inf, -np.inf], 0)

        # One-Hot Encoding for remaining categorical text columns
        X = pd.get_dummies(X)
        
        # Ensure Target is string
        y = y.astype(str)

        if len(y.unique()) < 2:
             raise HTTPException(status_code=400, detail="Target variable needs at least 2 different classes (e.g., 'Yes' and 'No').")

        # 4. Train Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if len(X_train) == 0:
             raise HTTPException(status_code=400, detail="Not enough data to split for training.")

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # 5. Results
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        importances = clf.feature_importances_
        feature_names = X.columns
        feature_imp_list = [
            {"name": str(name), "value": float(imp)} 
            for name, imp in zip(feature_names, importances)
        ]
        feature_imp_list.sort(key=lambda x: x['value'], reverse=True)

        return {
            "status": "success",
            "metrics": {
                "accuracy": float(acc)
            },
            "feature_importance": feature_imp_list,
            "confusion_matrix": cm.tolist(),
            "warning": warning_message,
            "features_used": available_features,
            "features_skipped": list(missing_features) if missing_features else []
        }

    except Exception as e:
        # This print statement will show up in your Terminal where uvicorn is running
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/statistics")
def get_statistics(request: TrainRequest):
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
        
        # Value counts for categorical columns (top 10)
        categorical_stats = {}
        for col in categorical_cols:
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