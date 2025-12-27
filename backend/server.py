from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, silhouette_score
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.inspection import permutation_importance

try:
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_APRIORI = True
except ImportError:
    HAS_APRIORI = False
    print("Warning: mlxtend not found. Apriori analysis will not be available.")

app = FastAPI()

# Global storage for trained model
model_artifact = {}

# Get environment
environment = os.getenv('ENVIRONMENT', 'development')

# CORS - allow local development, Docker, and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",           # Local React dev server
        "http://localhost:3000",           # Docker frontend (local)
        "http://127.0.0.1:3000",           # Docker frontend (local alt)
        "https://a2psdm.com",              # Production domain
        "https://www.a2psdm.com",          # Production domain with www
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

@app.post("/train-lr")
def train_logistic_regression(request: TrainRequest):
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

        # 4. Check Classification Applicability
        # Logistic Regression is strictly for Classification
        if pd.api.types.is_numeric_dtype(y):
             # If float values, it's regression -> Error for Logistic Regression
            if pd.api.types.is_float_dtype(y):
                 raise HTTPException(status_code=400, detail="Target variable appears to be continuous (regression). Logistic Regression is for classification only.")
            
            unique_count = len(y.unique())
            if unique_count > 20:
                 raise HTTPException(status_code=400, detail="Target variable has too many unique values for classification. Did you mean to use Regression?")
        
        y = y.astype(str) # Force string for classification
        
        if len(y.unique()) < 2:
            raise HTTPException(status_code=400, detail="Target variable needs at least 2 different classes.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Train Model
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # 6. Metrics
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
            "type": "classification"
        }

        # 7. Feature Importance (Coefficients)
        # For multi-class, coef_ is (n_classes, n_features). We take mean absolute value across classes.
        # For binary, coef_ is (1, n_features).
        if clf.coef_.ndim == 2:
             importances = np.mean(np.abs(clf.coef_), axis=0)
        else:
             importances = np.abs(clf.coef_).ravel()
             
        feature_imp_list = [
            {"name": str(name), "value": float(imp)} 
            for name, imp in zip(X.columns, importances)
        ]
        feature_imp_list.sort(key=lambda x: x['value'], reverse=True)

        return {
            "status": "success",
            "model_type": "classification",
            "metrics": metrics,
            "feature_importance": feature_imp_list,
            "confusion_matrix": cm_list,
            "warning": warning_message,
            "features_used": available_features,
            "features_skipped": list(missing_features) if missing_features else []
        }

    except Exception as e:
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-dt")
def train_decision_tree(request: TrainRequest):
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
        is_regression = False
        
        if pd.api.types.is_numeric_dtype(y):
            unique_count = len(y.unique())
            if pd.api.types.is_float_dtype(y):
                is_regression = True
            elif unique_count > 20:
                is_regression = True
            else:
                is_regression = False
        else:
            is_regression = False

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        metrics = {}
        cm_list = []
        
        if is_regression:
            # REGRESSION
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(0)
            
            clf = DecisionTreeRegressor(random_state=42)
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
            y = y.astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if len(y.unique()) < 2:
                raise HTTPException(status_code=400, detail="Target variable needs at least 2 different classes.")

            clf = DecisionTreeClassifier(random_state=42)
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
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-nn")
def train_neural_network(request: TrainRequest):
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
        is_regression = False
        
        if pd.api.types.is_numeric_dtype(y):
            unique_count = len(y.unique())
            if pd.api.types.is_float_dtype(y):
                is_regression = True
            elif unique_count > 20:
                is_regression = True
            else:
                is_regression = False
        else:
            is_regression = False

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        metrics = {}
        cm_list = []
        
        if is_regression:
            # REGRESSION
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(0)
            
            # MLP Regressor
            clf = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
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
            y = y.astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if len(y.unique()) < 2:
                raise HTTPException(status_code=400, detail="Target variable needs at least 2 different classes.")

            # MLP Classifier
            clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
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

        # Calculate Permutation Importance
        # n_repeats=10 for stability, random_state for reproducibility
        # scoring needs to be appropriate for the problem
        scoring = 'r2' if is_regression else 'accuracy'
        
        perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, scoring=scoring)
        importances = perm_importance.importances_mean
        
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
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-apriori")
def train_apriori(request: TrainRequest):
    if not HAS_APRIORI:
        raise HTTPException(status_code=500, detail="Server Error: 'mlxtend' library is not installed. Please install it to use Association Rules.")

    try:
        # 1. Load Data
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")

        # 2. Extract Features (Transactions)
        # Apriori ignores Target usually, or treats it as just another item.
        # But our UI forces Target selection. We can include Target in the basket or ignore it.
        # Let's include Target + Features as the "Basket" items if they are categorical.
        
        # We need to construct a "Basket" dataframe where columns are items and values are 0/1 (True/False).
        # Strategy: Select all requested columns (Target + Features). 
        # Convert all to string/categorical and One-Hot Encode.
        
        # Preprocessing Strategy:
        # CASE A: Traditional "Wide" Dataset or "One-Hot" intent.
        # CASE B: "Transaction/Long" Dataset (e.g. InvoiceID, Item).
        
        # Heuristic: If we have exactly 1 Feature and 1 Target, assume it's Transaction Data (Case B).
        # We group by the Feature (e.g. InvoiceID) and collect Target (e.g. Items) into baskets.
        
        cols_to_use = request.features + [request.target]
        # Filter for existing columns
        cols_to_use = [c for c in cols_to_use if c in df.columns]
        
        if not cols_to_use:
             raise HTTPException(status_code=400, detail="No valid columns selected for Association Rules.")

        df_subset = df[cols_to_use].copy()
        for col in df_subset.columns:
            df_subset[col] = df_subset[col].astype(str)
        
        basket = pd.DataFrame() # Placeholder
        
        # Check if we should Pivot (Market Basket Mode)
        # Condition: 1 Feature (the ID) and 1 Target (the Item)
        is_market_basket_mode = (len(request.features) == 1)
        
        if is_market_basket_mode:
            group_col = request.features[0]
            item_col = request.target
            
            # Group by ID and list items
            transactions = df_subset.groupby(group_col)[item_col].apply(list).tolist()
            
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            basket = pd.DataFrame(te_ary, columns=te.columns_)
            
            # If basket is too small (e.g. only 1 item type), we can't find associations
            if len(basket.columns) < 2:
                 # Fallback to simple one-hot if pivoting failed to produce columns
                 is_market_basket_mode = False 
        
        if not is_market_basket_mode:
            # Standard One-Hot encoding of all selected columns
            # This treats each row as a transaction containing the values of its columns
            basket = pd.get_dummies(df_subset)
            # Ensure boolean
            basket = basket.map(lambda x: 1 if x else 0).astype(bool)
        
        # 3. Validation
        if basket.empty:
             raise HTTPException(status_code=400, detail="Data transformation failed using One-Hot Encoding.")

        if basket.empty:
             raise HTTPException(status_code=400, detail="Data transformation failed using One-Hot Encoding.")

        # Performance Optimization: Limit to Top 50 frequent items
        # If there are too many columns, FP-Growth can still be slow.
        if len(basket.columns) > 50:
            # Calculate frequency of each item
            item_counts = basket.sum(axis=0).sort_values(ascending=False)
            top_items = item_counts.head(50).index.tolist()
            basket = basket[top_items]
            
            msg = f"Performance: Analyzed top 50 most frequent items (out of {len(item_counts)})."
            if warning_message:
                warning_message += " " + msg
            else:
                warning_message = msg

        # 4. Run FP-Growth (Faster than Apriori)
        # We also limit max_len=4 to prevent creating massive itemsets that explode computation time
        min_support = 0.05
        
        # Helper to run algorithm safely
        def run_mining(support_val):
             return fpgrowth(basket, min_support=support_val, use_colnames=True, max_len=4)

        frequent_itemsets = run_mining(min_support)
        
        if frequent_itemsets.empty:
            min_support = 0.01
            frequent_itemsets = run_mining(min_support)
            
        if frequent_itemsets.empty:
            min_support = 0.005
            frequent_itemsets = run_mining(min_support)

        if frequent_itemsets.empty:
             raise HTTPException(status_code=400, detail=f"No frequent itemsets found even with min_support={min_support}. Try evaluating a larger dataset or different columns.")

        # 5. Generate Rules
        # Relaxed Lift threshold to ensure we return SOMETHING, even if weak
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.001)
        
        if rules.empty:
             # Try confidence if lift fails? Usually lift > 0.001 should catch everything unless empty.
             raise HTTPException(status_code=400, detail="No association rules could be generated from the frequent itemsets.")

        # 6. Sort by Lift
        rules = rules.sort_values(by="lift", ascending=False)
        
        # 7. Format Output
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        
        # Select top 50 rules
        top_rules = rules.head(50)
        
        association_rules_list = []
        for _, row in top_rules.iterrows():
            association_rules_list.append({
                "antecedents": str(row["antecedents"]),
                "consequents": str(row["consequents"]),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"])
            })

        return {
            "status": "success",
            "model_type": "association_rules",
            "association_rules": association_rules_list,
            # Dummy values for other fields to satisfy frontend types if needed, or handle in frontend
            "metrics": {},
            "feature_importance": [],
            "confusion_matrix": [],
            "features_used": cols_to_use,
            "features_skipped": []
        }

    except Exception as e:
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-pca")
def train_pca(request: TrainRequest):
    try:
        # 1. Load Data
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")

        # 2. Select Features
        # PCA requires numerical data.
        cols_to_use = request.features
        # If target is selected and is numeric, we can optionally include it, but usually PCA is unsupervised.
        # For now we just use the selected features.
        
        cols_to_use = [c for c in cols_to_use if c in df.columns]
        
        if not cols_to_use:
             raise HTTPException(status_code=400, detail="No valid columns selected for PCA.")

        df_subset = df[cols_to_use].copy()
        
        # 3. Preprocessing
        # Convert to numeric, coerce errors
        for col in df_subset.columns:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
            
        # Drop rows with NaN (or fill?) - dropping is safer for PCA consistency usually, but filling is more robust for user.
        # Let's fill with mean
        df_subset = df_subset.fillna(df_subset.mean())
        
        if df_subset.empty:
             raise HTTPException(status_code=400, detail="Data validation failed (no valid numeric data remaining).")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_subset)
        
        # 4. Run PCA
        # We compute all components to show variance curve
        n_components = min(len(cols_to_use), 10) # Limit to 10 or num features
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)
        
        # 5. Prepare Results
        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        
        # Components (Loadings) - Shape: [n_components, n_features]
        # We'll map them to feature names
        components_list = []
        for i, comp in enumerate(pca.components_):
            comp_dict = {"component": f"PC{i+1}"}
            for j, val in enumerate(comp):
                comp_dict[cols_to_use[j]] = float(val)
            components_list.append(comp_dict)
            
        # Transformed Data (Projected) - First 2 components for scatter plot
        # We associate a label if target is provided (for coloring)
        projected_data = []
        labels = None
        if request.target and request.target in df.columns:
            labels = df[request.target].astype(str).tolist()
        else:
            labels = [""] * len(df)
            
        # Limit scatter points to 500 to prevent lag
        limit = min(len(pca_result), 500)
        for i in range(limit):
            projected_data.append({
                "x": float(pca_result[i, 0]) if n_components > 0 else 0,
                "y": float(pca_result[i, 1]) if n_components > 1 else 0,
                "label": str(labels[i])
            })

        return {
            "status": "success",
            "model_type": "pca",
            "pca_results": {
                "explained_variance": explained_variance_ratio,
                "components": components_list,
                "projected_data": projected_data
            },
            "metrics": {},
            "feature_importance": [],
            "confusion_matrix": [],
            "features_used": cols_to_use,
            "features_skipped": []
        }

    except Exception as e:
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-fa")
def train_fa(request: TrainRequest):
    try:
        # 1. Load Data
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")

        # 2. Select Features
        # FA requires numerical data.
        cols_to_use = request.features
        
        cols_to_use = [c for c in cols_to_use if c in df.columns]
        
        if not cols_to_use:
             raise HTTPException(status_code=400, detail="No valid columns selected for Factor Analysis.")

        df_subset = df[cols_to_use].copy()
        
        # 3. Preprocessing
        # Convert to numeric, coerce errors
        for col in df_subset.columns:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
            
        # Fill missing with mean
        df_subset = df_subset.fillna(df_subset.mean())
        
        if df_subset.empty:
             raise HTTPException(status_code=400, detail="Data validation failed (no valid numeric data remaining).")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_subset)
        
        # 4. Run Factor Analysis
        # Limit components to min(features, 10)
        n_components = min(len(cols_to_use), 10)
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        fa.fit(X_scaled)
        
        # 5. Prepare Results
        # Loadings: fa.components_ (n_components, n_features)
        components_list = []
        for i, comp in enumerate(fa.components_):
            comp_dict = {"component": f"Factor{i+1}"}
            # Calculate sum of squared loadings for this factor (Eigenvalue-ish)
            ss_loadings = sum(c**2 for c in comp)
            comp_dict["variance"] = float(ss_loadings)
            
            for j, val in enumerate(comp):
                comp_dict[cols_to_use[j]] = float(val)
            components_list.append(comp_dict)
            
        # Communalities (Sum of squared loadings for each feature)
        # fa.components_ is (n_components, n_features)
        # We want sum over components for each feature
        communalities = np.sum(fa.components_**2, axis=0).tolist()
        communalities_dict = {feat: float(val) for feat, val in zip(cols_to_use, communalities)}

        return {
            "status": "success",
            "model_type": "factor-analysis",
            "fa_results": {
                "components": components_list,
                "communalities": communalities_dict,
                "noise_variance": fa.noise_variance_.tolist() if hasattr(fa, 'noise_variance_') else [],
                "log_likelihood": float(fa.score(X_scaled))
            },
            "metrics": {},
            "feature_importance": [],
            "confusion_matrix": [],
            "features_used": cols_to_use,
            "features_skipped": []
        }

    except Exception as e:
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-kmeans")
def train_kmeans(request: TrainRequest):
    try:
        # 1. Load Data
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")

        # 2. Select Features
        cols_to_use = request.features
        cols_to_use = [c for c in cols_to_use if c in df.columns]
        
        if not cols_to_use:
             raise HTTPException(status_code=400, detail="No valid columns selected for K-Means.")

        df_subset = df[cols_to_use].copy()
        
        # 3. Preprocessing
        for col in df_subset.columns:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
        
        # Drop columns that are entirely NaN
        df_subset = df_subset.dropna(axis=1, how='all')
        
        # Fill remaining NaN with column mean, then 0 as fallback
        df_subset = df_subset.fillna(df_subset.mean())
        df_subset = df_subset.fillna(0)
        
        # Drop any rows that still have NaN (edge case)
        df_subset = df_subset.dropna()
        
        if df_subset.empty or len(df_subset) < 3:
             raise HTTPException(status_code=400, detail="Not enough valid data for K-Means. Need at least 3 data points.")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_subset)
        
        # 4. Run K-Means
        n_clusters = 3  # Default
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate Silhouette Score
        sil_score = float(silhouette_score(X_scaled, cluster_labels)) if len(set(cluster_labels)) > 1 else 0.0
        
        # Generate Elbow Plot data (K=2 to 6)
        elbow_data = []
        for k in range(2, min(7, len(df_subset))):
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_temp.fit(X_scaled)
            elbow_data.append({"k": k, "inertia": float(km_temp.inertia_)})
        
        # 5. Prepare Results
        # Cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_distribution = [{"cluster": int(u), "count": int(c)} for u, c in zip(unique, counts)]
        
        # Cluster centers (inverse transform to original scale)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_list = []
        for i, center in enumerate(centers):
            center_dict = {"cluster": i}
            for j, val in enumerate(center):
                center_dict[cols_to_use[j]] = float(val)
            centers_list.append(center_dict)

        return {
            "status": "success",
            "model_type": "kmeans",
            "kmeans_results": {
                "cluster_distribution": cluster_distribution,
                "cluster_centers": centers_list,
                "inertia": float(kmeans.inertia_),
                "n_clusters": n_clusters,
                "silhouette_score": sil_score,
                "elbow_data": elbow_data
            },
            "metrics": {},
            "feature_importance": [],
            "confusion_matrix": [],
            "features_used": cols_to_use,
            "features_skipped": []
        }

    except Exception as e:
        print(f"INTERNAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

@app.post("/train-hierarchical")
def train_hierarchical(request: TrainRequest):
    try:
        # 1. Load Data
        df = pd.DataFrame(request.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")

        # 2. Select Features
        cols_to_use = [c for c in request.features if c in df.columns]
        if not cols_to_use:
             raise HTTPException(status_code=400, detail="No valid columns selected.")

        df_subset = df[cols_to_use].copy()
        
        # 3. Preprocessing
        for col in df_subset.columns:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
        df_subset = df_subset.dropna(axis=1, how='all')
        df_subset = df_subset.fillna(df_subset.mean()).fillna(0)
        df_subset = df_subset.dropna()
        
        if df_subset.empty or len(df_subset) < 3:
             raise HTTPException(status_code=400, detail="Not enough valid data. Need at least 3 data points.")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_subset)
        
        # 4. Run Hierarchical Clustering
        n_clusters = 3
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = hc.fit_predict(X_scaled)
        
        # 5. Prepare Results
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_distribution = [{"cluster": int(u), "count": int(c)} for u, c in zip(unique, counts)]
        
        # Cluster centers (calculate mean for each cluster)
        centers_list = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            if mask.sum() > 0:
                center = scaler.inverse_transform(X_scaled[mask].mean(axis=0).reshape(1, -1))[0]
                center_dict = {"cluster": i}
                for j, val in enumerate(center):
                    center_dict[df_subset.columns[j]] = float(val)
                centers_list.append(center_dict)

        return {
            "status": "success",
            "model_type": "hierarchical",
            "hierarchical_results": {
                "cluster_distribution": cluster_distribution,
                "cluster_centers": centers_list,
                "n_clusters": n_clusters,
                "linkage": "ward"
            },
            "metrics": {},
            "feature_importance": [],
            "confusion_matrix": [],
            "features_used": list(df_subset.columns),
            "features_skipped": []
        }

    except Exception as e:
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

class ParetoRequest(BaseModel):
    data: List[Dict[str, Any]]
    category_column: str
    value_column: str

@app.post("/pareto-analysis")
def pareto_analysis(request: ParetoRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        cat_col = request.category_column
        val_col = request.value_column
        
        if cat_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Category column '{cat_col}' not found.")
        if val_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Value column '{val_col}' not found.")
        
        # Convert value column to numeric
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[val_col])
        
        # Group by category and sum values
        grouped = df.groupby(cat_col)[val_col].sum().reset_index()
        grouped.columns = ['category', 'value']
        
        # Sort descending
        grouped = grouped.sort_values('value', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        total = grouped['value'].sum()
        grouped['percentage'] = (grouped['value'] / total * 100).round(2)
        grouped['cumulative_percentage'] = grouped['percentage'].cumsum().round(2)
        
        # Find 80% threshold index
        threshold_80_idx = (grouped['cumulative_percentage'] >= 80).idxmax() if (grouped['cumulative_percentage'] >= 80).any() else len(grouped) - 1
        
        pareto_data = grouped.to_dict('records')
        
        return {
            "status": "success",
            "pareto_data": pareto_data,
            "total": float(total),
            "threshold_80_index": int(threshold_80_idx),
            "categories_for_80": int(threshold_80_idx + 1),
            "total_categories": len(grouped)
        }
        
    except Exception as e:
        print(f"PARETO ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

class HypothesisTestRequest(BaseModel):
    data: List[Dict[str, Any]]
    test_type: str  # 'one-sample', 'two-sample', 'paired'
    column1: str
    column2: Optional[str] = None  # Optional, for two-sample or paired
    hypothesized_mean: float = 0  # For one-sample t-test

@app.post("/hypothesis-test")
def hypothesis_test(request: HypothesisTestRequest):
    from scipy import stats
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        col1 = request.column1
        if col1 not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col1}' not found.")
        
        # Convert to numeric
        sample1 = pd.to_numeric(df[col1], errors='coerce').dropna()
        
        if len(sample1) < 2:
            raise HTTPException(status_code=400, detail="Not enough data points for testing.")
        
        result = {}
        
        if request.test_type == 'one-sample':
            # One-sample t-test against hypothesized mean
            t_stat, p_value = stats.ttest_1samp(sample1, request.hypothesized_mean)
            # Convert numpy types to Python native types
            t_stat = float(t_stat) if hasattr(t_stat, 'item') else float(t_stat)
            p_value = float(p_value) if hasattr(p_value, 'item') else float(p_value)
            result = {
                "test_type": "One-Sample T-Test",
                "test_type_zh": "單樣本 T 檢定",
                "sample_mean": float(sample1.mean()),
                "sample_std": float(sample1.std()),
                "sample_size": int(len(sample1)),
                "hypothesized_mean": float(request.hypothesized_mean),
                "t_statistic": t_stat,
                "p_value": p_value,
                "degrees_of_freedom": int(len(sample1) - 1),
                "significant_005": bool(p_value < 0.05),
                "significant_001": bool(p_value < 0.01)
            }
        
        elif request.test_type == 'two-sample':
            # Two-sample independent t-test (stack format: Y=numeric, X=categorical group)
            if not request.column2 or request.column2 not in df.columns:
                raise HTTPException(status_code=400, detail="Group column required for two-sample test.")
            
            # column1 = numeric values (Y), column2 = categorical groups (X)
            df_subset = df[[col1, request.column2]].dropna()
            df_subset[col1] = pd.to_numeric(df_subset[col1], errors='coerce')
            df_subset = df_subset.dropna()
            
            # Get unique groups
            groups = df_subset[request.column2].unique()
            if len(groups) < 2:
                raise HTTPException(status_code=400, detail=f"Need at least 2 groups in '{request.column2}'. Found: {len(groups)}")
            
            # Use first two groups for comparison
            group1_name = str(groups[0])
            group2_name = str(groups[1])
            sample1 = df_subset[df_subset[request.column2] == groups[0]][col1]
            sample2 = df_subset[df_subset[request.column2] == groups[1]][col1]
            
            if len(sample1) < 2 or len(sample2) < 2:
                raise HTTPException(status_code=400, detail="Not enough data points in one or both groups.")
            
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            t_stat = float(t_stat) if hasattr(t_stat, 'item') else float(t_stat)
            p_value = float(p_value) if hasattr(p_value, 'item') else float(p_value)
            result = {
                "test_type": "Two-Sample T-Test (Independent)",
                "test_type_zh": "雙樣本 T 檢定（獨立）",
                "group1_name": group1_name,
                "group2_name": group2_name,
                "sample1_mean": float(sample1.mean()),
                "sample1_std": float(sample1.std()),
                "sample1_size": int(len(sample1)),
                "sample2_mean": float(sample2.mean()),
                "sample2_std": float(sample2.std()),
                "sample2_size": int(len(sample2)),
                "mean_difference": float(sample1.mean() - sample2.mean()),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_005": bool(p_value < 0.05),
                "significant_001": bool(p_value < 0.01)
            }
        
        elif request.test_type == 'paired':
            # Paired t-test
            if not request.column2 or request.column2 not in df.columns:
                raise HTTPException(status_code=400, detail="Second column required for paired test.")
            
            # Get both samples fresh
            sample1_paired = pd.to_numeric(df[col1], errors='coerce')
            sample2_paired = pd.to_numeric(df[request.column2], errors='coerce')
            
            # Create paired DataFrame and drop rows with any NaN
            paired_df = pd.DataFrame({'s1': sample1_paired, 's2': sample2_paired}).dropna()
            
            if len(paired_df) < 2:
                raise HTTPException(status_code=400, detail="Not enough paired data points.")
            
            s1 = paired_df['s1']
            s2 = paired_df['s2']
            
            t_stat, p_value = stats.ttest_rel(s1, s2)
            t_stat = float(t_stat) if hasattr(t_stat, 'item') else float(t_stat)
            p_value = float(p_value) if hasattr(p_value, 'item') else float(p_value)
            diff = s1.values - s2.values
            result = {
                "test_type": "Paired T-Test",
                "test_type_zh": "配對 T 檢定",
                "sample1_mean": float(s1.mean()),
                "sample2_mean": float(s2.mean()),
                "mean_difference": float(diff.mean()),
                "std_difference": float(diff.std()),
                "paired_size": int(len(paired_df)),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_005": bool(p_value < 0.05),
                "significant_001": bool(p_value < 0.01)
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown test type: {request.test_type}")
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        print(f"HYPOTHESIS TEST ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

class ChiSquareRequest(BaseModel):
    data: List[Dict[str, Any]]
    column1: str
    column2: str

@app.post("/chi-square-test")
def chi_square_test(request: ChiSquareRequest):
    from scipy import stats
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        col1 = request.column1
        col2 = request.column2
        
        if col1 not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col1}' not found.")
        if col2 not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col2}' not found.")
        
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Not enough categories for Chi-Square test. Need at least 2x2 table.")
        
        # Perform Chi-Square test
        chi2, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        
        # Create readable table
        observed_table = contingency_table.to_dict('index')
        expected_table = pd.DataFrame(
            expected_freq, 
            index=contingency_table.index, 
            columns=contingency_table.columns
        ).round(2).to_dict('index')
        
        # Calculate Cramér's V (effect size)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            "status": "success",
            "chi_square_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "cramers_v": float(cramers_v),
            "significant_005": bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.01),
            "observed_table": observed_table,
            "expected_table": expected_table,
            "row_categories": [str(x) for x in contingency_table.index.tolist()],
            "col_categories": [str(x) for x in contingency_table.columns.tolist()],
            "table_shape": list(contingency_table.shape)
        }
        
    except Exception as e:
        print(f"CHI-SQUARE ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

class RegressionRequest(BaseModel):
    data: List[Dict[str, Any]]
    x_column: str
    y_column: str

@app.post("/regression-analysis")
def regression_analysis(request: RegressionRequest):
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        x_col = request.x_column
        y_col = request.y_column
        
        if x_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{x_col}' not found.")
        if y_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{y_col}' not found.")
        
        # Convert to numeric and clean
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df_clean = df[[x_col, y_col]].dropna()
        
        if len(df_clean) < 3:
            raise HTTPException(status_code=400, detail="Not enough data points for regression. Need at least 3.")
        
        X = df_clean[x_col].values.reshape(-1, 1)
        y = df_clean[y_col].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Calculate R² and adjusted R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        n = len(y)
        p = 1  # number of predictors
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        
        # Calculate standard error and p-value for slope
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # Standard error of the estimate
        se_estimate = np.sqrt(ss_res / (n - 2)) if n > 2 else 0
        
        # Standard error of slope
        x_mean = np.mean(X)
        ss_x = np.sum((X.flatten() - x_mean) ** 2)
        se_slope = se_estimate / np.sqrt(ss_x) if ss_x > 0 else 0
        
        # T-statistic and p-value for slope
        t_stat = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1
        
        # Correlation coefficient
        correlation = np.corrcoef(X.flatten(), y)[0, 1]
        
        # Prepare scatter data (sample if too large)
        scatter_data = []
        sample_size = min(200, len(df_clean))
        sample_indices = np.random.choice(len(df_clean), sample_size, replace=False) if len(df_clean) > sample_size else range(len(df_clean))
        for i in sample_indices:
            scatter_data.append({
                "x": float(df_clean.iloc[i][x_col]),
                "y": float(df_clean.iloc[i][y_col])
            })
        
        # Regression line endpoints
        x_min, x_max = float(df_clean[x_col].min()), float(df_clean[x_col].max())
        
        return {
            "status": "success",
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "se_slope": float(se_slope),
            "se_estimate": float(se_estimate),
            "n_observations": int(n),
            "significant_005": bool(p_value < 0.05),
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "scatter_data": scatter_data,
            "line_start": {"x": float(x_min), "y": float(slope * x_min + intercept)},
            "line_end": {"x": float(x_max), "y": float(slope * x_max + intercept)}
        }
        
    except Exception as e:
        print(f"REGRESSION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

class MultipleRegressionRequest(BaseModel):
    data: List[Dict[str, Any]]
    x_columns: List[str]
    y_column: str

@app.post("/multiple-regression")
def multiple_regression(request: MultipleRegressionRequest):
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        y_col = request.y_column
        x_cols = request.x_columns
        
        if y_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{y_col}' not found.")
        
        missing = [c for c in x_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
        
        if len(x_cols) < 1:
            raise HTTPException(status_code=400, detail="Need at least 1 predictor column.")
        
        # Convert to numeric and clean
        for col in x_cols + [y_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_clean = df[x_cols + [y_col]].dropna()
        
        if len(df_clean) < len(x_cols) + 2:
            raise HTTPException(status_code=400, detail="Not enough data points for regression.")
        
        X = df_clean[x_cols].values
        y = df_clean[y_col].values
        
        # Fit multiple regression
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Calculate R² and adjusted R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        n = len(y)
        p = len(x_cols)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        
        # Coefficients
        coefficients = []
        for i, col in enumerate(x_cols):
            coefficients.append({
                "variable": col,
                "coefficient": float(model.coef_[i])
            })
        
        # F-statistic and p-value
        if p > 0 and n > p + 1 and ss_res > 0:
            msr = (ss_tot - ss_res) / p  # Mean Square Regression
            mse = ss_res / (n - p - 1)   # Mean Square Error
            f_stat = msr / mse if mse > 0 else 0
            f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        else:
            f_stat = 0
            f_p_value = 1
        
        return {
            "status": "success",
            "intercept": float(model.intercept_),
            "coefficients": coefficients,
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "f_statistic": float(f_stat),
            "f_p_value": float(f_p_value),
            "n_observations": int(n),
            "n_predictors": int(p),
            "significant_005": bool(f_p_value < 0.05)
        }
        
    except Exception as e:
        print(f"MULTIPLE REGRESSION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

class CorrelationMatrixRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str] = None  # Optional, if None use all numeric columns

@app.post("/correlation-matrix")
def correlation_matrix(request: CorrelationMatrixRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        # Select columns
        if request.columns:
            missing = [c for c in request.columns if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
            cols = request.columns
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(cols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns.")
        
        # Convert to numeric and keep only valid numeric columns
        df_numeric = pd.DataFrame()
        for col in cols:
            converted = pd.to_numeric(df[col], errors='coerce')
            # Only keep columns where at least 3 values are numeric
            if converted.notna().sum() >= 3:
                df_numeric[col] = converted
        
        if len(df_numeric.columns) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 columns with numeric data.")
        
        if len(df_numeric.dropna()) < 3:
            # Use pairwise correlation if too many missing values
            df_subset = df_numeric
        else:
            df_subset = df_numeric
        
        # Compute correlation matrix
        corr_matrix = df_subset.corr()
        
        # Convert to list format for frontend
        matrix_data = []
        for i, row_name in enumerate(corr_matrix.index):
            for j, col_name in enumerate(corr_matrix.columns):
                matrix_data.append({
                    "row": str(row_name),
                    "col": str(col_name),
                    "value": round(float(corr_matrix.iloc[i, j]), 4)
                })
        
        return {
            "status": "success",
            "columns": [str(c) for c in corr_matrix.columns.tolist()],
            "matrix_data": matrix_data,
            "n_observations": int(len(df_subset))
        }
        
    except Exception as e:
        print(f"CORRELATION MATRIX ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

# Anomaly Detection
class AnomalyDetectionRequest(BaseModel):
    data: List[Dict[str, Any]]
    column: str
    method: str = "zscore"  # 'zscore' or 'iqr'
    threshold: float = 3.0  # Z-score threshold or IQR multiplier

@app.post("/anomaly-detection")
def anomaly_detection(request: AnomalyDetectionRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        col = request.column
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
        
        # Convert to numeric
        values = pd.to_numeric(df[col], errors='coerce')
        valid_mask = values.notna()
        valid_values = values[valid_mask]
        
        if len(valid_values) < 3:
            raise HTTPException(status_code=400, detail="Not enough numeric data for anomaly detection.")
        
        mean_val = float(valid_values.mean())
        std_val = float(valid_values.std())
        q1 = float(valid_values.quantile(0.25))
        q3 = float(valid_values.quantile(0.75))
        iqr = q3 - q1
        
        anomalies = []
        
        if request.method == "zscore":
            # Z-score method
            if std_val > 0:
                z_scores = (valid_values - mean_val) / std_val
                anomaly_mask = abs(z_scores) > request.threshold
            else:
                anomaly_mask = pd.Series([False] * len(valid_values), index=valid_values.index)
            
            lower_bound = mean_val - request.threshold * std_val
            upper_bound = mean_val + request.threshold * std_val
            
        else:  # IQR method
            lower_bound = q1 - request.threshold * iqr
            upper_bound = q3 + request.threshold * iqr
            anomaly_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
        
        # Collect anomaly details
        anomaly_indices = valid_values[anomaly_mask].index.tolist()
        for idx in anomaly_indices:
            row_data = df.iloc[idx].to_dict()
            anomalies.append({
                "row_index": int(idx),
                "value": float(values.iloc[idx]),
                "row_data": {k: str(v) for k, v in row_data.items()}
            })
        
        return {
            "status": "success",
            "column": col,
            "method": request.method,
            "threshold": float(request.threshold),
            "total_rows": int(len(valid_values)),
            "anomaly_count": int(len(anomalies)),
            "anomaly_percentage": round(float(len(anomalies) / len(valid_values) * 100), 2) if len(valid_values) > 0 else 0,
            "statistics": {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4)
            },
            "anomalies": anomalies[:100]  # Limit to first 100 anomalies
        }
        
    except Exception as e:
        print(f"ANOMALY DETECTION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

# Word Cloud - Word Frequency Extraction
class WordCloudRequest(BaseModel):
    data: List[Dict[str, Any]]
    column: str
    max_words: int = 100
    min_frequency: int = 1

@app.post("/word-cloud")
def word_cloud(request: WordCloudRequest):
    import re
    from collections import Counter
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        col = request.column
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
        
        # Collect all text from the column
        all_text = df[col].dropna().astype(str).str.cat(sep=' ')
        
        if not all_text.strip():
            raise HTTPException(status_code=400, detail="No text data found in the column.")
        
        # Extract words (handles both Chinese and English)
        # For Chinese: split by common delimiters and get individual characters/words
        # For English: extract words
        
        # Remove common punctuation and split
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', all_text)
        
        # Split into words/characters
        words = cleaned_text.split()
        
        # Also handle Chinese characters individually if they appear
        expanded_words = []
        for word in words:
            if re.match(r'^[\u4e00-\u9fff]+$', word):
                # Chinese text - can keep as word or split into characters
                # Keep as single word/phrase if short, otherwise keep whole
                if len(word) <= 4:
                    expanded_words.append(word)
                else:
                    # Split long Chinese text into 2-char chunks
                    for i in range(0, len(word), 2):
                        chunk = word[i:i+2]
                        if len(chunk) >= 1:
                            expanded_words.append(chunk)
            else:
                # English or other - convert to lowercase
                expanded_words.append(word.lower())
        
        # Filter short words and count frequencies
        filtered_words = [w for w in expanded_words if len(w) >= 1]
        word_counts = Counter(filtered_words)
        
        # Filter by minimum frequency and get top words
        frequent_words = [(word, count) for word, count in word_counts.most_common() 
                         if count >= request.min_frequency][:request.max_words]
        
        if not frequent_words:
            raise HTTPException(status_code=400, detail="No words found with minimum frequency.")
        
        # Calculate max count for normalization
        max_count = frequent_words[0][1] if frequent_words else 1
        
        # Format for frontend
        words_data = [
            {
                "text": word,
                "count": int(count),
                "size": max(12, min(60, int(20 + 40 * (count / max_count))))  # Size between 12-60px
            }
            for word, count in frequent_words
        ]
        
        return {
            "status": "success",
            "column": col,
            "total_words": int(len(filtered_words)),
            "unique_words": int(len(word_counts)),
            "words": words_data
        }
        
    except Exception as e:
        print(f"WORD CLOUD ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

# Holt-Winters Time Series Forecasting
class HoltWintersRequest(BaseModel):
    data: List[Dict[str, Any]]
    value_column: str
    date_column: Optional[str] = None
    seasonal_periods: int = 12
    forecast_periods: int = 6
    trend: Optional[str] = "add"  # 'add', 'mul', or None
    seasonal: Optional[str] = "add"  # 'add', 'mul', or None

@app.post("/holt-winters")
def holt_winters_forecast(request: HoltWintersRequest):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        val_col = request.value_column
        if val_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{val_col}' not found.")
        
        # Extract values
        values = pd.to_numeric(df[val_col], errors='coerce').dropna()
        
        if len(values) < 2 * request.seasonal_periods:
            raise HTTPException(status_code=400, detail=f"Need at least {2 * request.seasonal_periods} data points for seasonal analysis.")
        
        # Reset index for proper time series
        values = values.reset_index(drop=True)
        
        # Prepare trend and seasonal parameters
        trend_type = request.trend if request.trend in ['add', 'mul'] else None
        seasonal_type = request.seasonal if request.seasonal in ['add', 'mul'] else None
        
        # Fit Holt-Winters model
        try:
            model = ExponentialSmoothing(
                values,
                trend=trend_type,
                seasonal=seasonal_type,
                seasonal_periods=request.seasonal_periods,
                initialization_method='estimated'
            )
            fitted = model.fit(optimized=True)
        except Exception as fit_error:
            # Fallback to simpler model if complex fails
            model = ExponentialSmoothing(
                values,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fitted = model.fit(optimized=True)
        
        # Generate forecast
        forecast = fitted.forecast(request.forecast_periods)
        
        # Get fitted values
        fitted_values = fitted.fittedvalues
        
        # Calculate metrics
        residuals = values - fitted_values
        mae = float(abs(residuals).mean())
        mse = float((residuals ** 2).mean())
        rmse = float(np.sqrt(mse))
        mape = float((abs(residuals / values) * 100).mean()) if (values != 0).all() else None
        
        # Prepare historical data
        historical = [
            {"index": int(i), "actual": float(values.iloc[i]), "fitted": float(fitted_values.iloc[i])}
            for i in range(len(values))
        ]
        
        # Prepare forecast data
        forecast_data = [
            {"index": int(len(values) + i), "forecast": float(forecast.iloc[i])}
            for i in range(len(forecast))
        ]
        
        # Get model parameters
        params = {
            "alpha": float(fitted.params.get('smoothing_level', 0)),
            "beta": float(fitted.params.get('smoothing_trend', 0)) if fitted.params.get('smoothing_trend') else None,
            "gamma": float(fitted.params.get('smoothing_seasonal', 0)) if fitted.params.get('smoothing_seasonal') else None,
        }
        
        return {
            "status": "success",
            "column": val_col,
            "total_observations": int(len(values)),
            "forecast_periods": request.forecast_periods,
            "seasonal_periods": request.seasonal_periods,
            "trend_type": trend_type or "none",
            "seasonal_type": seasonal_type or "none",
            "metrics": {
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 2) if mape else None,
                "aic": round(float(fitted.aic), 2) if hasattr(fitted, 'aic') else None,
                "bic": round(float(fitted.bic), 2) if hasattr(fitted, 'bic') else None
            },
            "parameters": params,
            "historical": historical,
            "forecast": forecast_data
        }
        
    except Exception as e:
        print(f"HOLT-WINTERS ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

# ARIMA Time Series Forecasting
class ARIMARequest(BaseModel):
    data: List[Dict[str, Any]]
    value_column: str
    p: int = 1  # AR order
    d: int = 1  # Differencing
    q: int = 1  # MA order
    forecast_periods: int = 6

@app.post("/arima")
def arima_forecast(request: ARIMARequest):
    from statsmodels.tsa.arima.model import ARIMA
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        val_col = request.value_column
        if val_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{val_col}' not found.")
        
        # Extract values
        values = pd.to_numeric(df[val_col], errors='coerce').dropna()
        
        min_samples = request.p + request.d + request.q + 10
        if len(values) < min_samples:
            raise HTTPException(status_code=400, detail=f"Need at least {min_samples} data points for ARIMA({request.p},{request.d},{request.q}).")
        
        # Reset index for proper time series
        values = values.reset_index(drop=True)
        
        # Fit ARIMA model
        try:
            model = ARIMA(values, order=(request.p, request.d, request.q))
            fitted = model.fit()
        except Exception as fit_error:
            print(f"ARIMA fit error: {fit_error}")
            # Try simpler model
            model = ARIMA(values, order=(1, 1, 0))
            fitted = model.fit()
        
        # Generate forecast with confidence intervals
        forecast_result = fitted.get_forecast(steps=request.forecast_periods)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)
        
        # Get fitted values
        fitted_values = fitted.fittedvalues
        
        # Calculate metrics
        # Start from d+1 since first d values may be NaN
        start_idx = request.d
        actual_trimmed = values.iloc[start_idx:]
        fitted_trimmed = fitted_values.iloc[start_idx:]
        
        residuals = actual_trimmed - fitted_trimmed
        mae = float(abs(residuals).mean())
        mse = float((residuals ** 2).mean())
        rmse = float(np.sqrt(mse))
        mape = float((abs(residuals / actual_trimmed) * 100).mean()) if (actual_trimmed != 0).all() else None
        
        # Prepare historical data
        historical = []
        for i in range(len(values)):
            hist_item = {"index": int(i), "actual": float(values.iloc[i])}
            if i >= start_idx and i < len(fitted_values):
                hist_item["fitted"] = float(fitted_values.iloc[i])
            historical.append(hist_item)
        
        # Prepare forecast data
        forecast_data = []
        for i in range(len(forecast_mean)):
            forecast_data.append({
                "index": int(len(values) + i),
                "forecast": float(forecast_mean.iloc[i]),
                "lower_ci": float(conf_int.iloc[i, 0]),
                "upper_ci": float(conf_int.iloc[i, 1])
            })
        
        return {
            "status": "success",
            "column": val_col,
            "order": {"p": request.p, "d": request.d, "q": request.q},
            "total_observations": int(len(values)),
            "forecast_periods": request.forecast_periods,
            "metrics": {
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 2) if mape else None,
                "aic": round(float(fitted.aic), 2),
                "bic": round(float(fitted.bic), 2)
            },
            "historical": historical,
            "forecast": forecast_data
        }
        
    except Exception as e:
        print(f"ARIMA ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

# Croston's Method for Intermittent Demand
class CrostonRequest(BaseModel):
    data: List[Dict[str, Any]]
    value_column: str
    alpha: float = 0.1  # Smoothing parameter
    forecast_periods: int = 6

@app.post("/croston")
def croston_forecast(request: CrostonRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        val_col = request.value_column
        if val_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{val_col}' not found.")
        
        # Extract values
        values = pd.to_numeric(df[val_col], errors='coerce').fillna(0).values
        
        if len(values) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points for Croston's method.")
        
        alpha = request.alpha
        
        # Croston's method implementation
        # Separate demand sizes and intervals
        demand_times = []  # Indices where demand occurs
        demand_sizes = []  # Non-zero demand values
        
        for i, v in enumerate(values):
            if v > 0:
                demand_times.append(i)
                demand_sizes.append(v)
        
        if len(demand_sizes) < 3:
            raise HTTPException(status_code=400, detail="Not enough non-zero demands. Croston's method is for intermittent demand data.")
        
        # Calculate inter-demand intervals
        intervals = [demand_times[i] - demand_times[i-1] for i in range(1, len(demand_times))]
        
        # Initialize smoothed values
        z = demand_sizes[0]  # Smoothed demand size
        p = intervals[0] if intervals else 1  # Smoothed interval
        
        # Lists to track smoothed values
        z_values = [z]
        p_values = [p]
        
        # Apply exponential smoothing to demand sizes
        for i in range(1, len(demand_sizes)):
            z = alpha * demand_sizes[i] + (1 - alpha) * z
            z_values.append(z)
        
        # Apply exponential smoothing to intervals
        for i in range(1, len(intervals)):
            p = alpha * intervals[i] + (1 - alpha) * p
            p_values.append(p)
        
        # Final smoothed values
        final_z = z_values[-1]
        final_p = max(p_values[-1], 0.001)  # Avoid division by zero
        
        # Croston's forecast = z / p (demand rate per period)
        croston_forecast_value = final_z / final_p
        
        # Calculate fitted values
        fitted = []
        current_z = demand_sizes[0]
        current_p = intervals[0] if intervals else 1
        demand_idx = 0
        interval_idx = 0
        
        for i in range(len(values)):
            if i in demand_times and demand_idx < len(demand_sizes):
                if demand_idx > 0:
                    current_z = alpha * demand_sizes[demand_idx] + (1 - alpha) * current_z
                demand_idx += 1
            if demand_idx > 1 and interval_idx < len(intervals):
                current_p = alpha * intervals[interval_idx] + (1 - alpha) * current_p
                interval_idx += 1
            fitted_value = current_z / max(current_p, 0.001)
            fitted.append(fitted_value)
        
        # Calculate metrics
        actual_values = values
        fitted_values = np.array(fitted)
        residuals = actual_values - fitted_values
        mae = float(abs(residuals).mean())
        rmse = float(np.sqrt((residuals ** 2).mean()))
        # MAPE only for non-zero actual values
        nonzero_mask = actual_values > 0
        if nonzero_mask.sum() > 0:
            mape = float((abs(residuals[nonzero_mask] / actual_values[nonzero_mask]) * 100).mean())
        else:
            mape = None
        
        # Calculate demand statistics
        zero_count = int((values == 0).sum())
        nonzero_count = int((values > 0).sum())
        avg_demand = float(np.mean([v for v in values if v > 0])) if nonzero_count > 0 else 0
        avg_interval = float(np.mean(intervals)) if intervals else 0
        
        # Prepare historical data
        historical = [
            {"index": int(i), "actual": float(values[i]), "fitted": float(fitted[i])}
            for i in range(len(values))
        ]
        
        # Prepare forecast data (constant forecast for intermittent demand)
        forecast_data = [
            {"index": int(len(values) + i), "forecast": float(croston_forecast_value)}
            for i in range(request.forecast_periods)
        ]
        
        return {
            "status": "success",
            "column": val_col,
            "alpha": float(alpha),
            "total_observations": int(len(values)),
            "forecast_periods": request.forecast_periods,
            "demand_stats": {
                "zero_periods": zero_count,
                "nonzero_periods": nonzero_count,
                "intermittence_ratio": round(float(zero_count / len(values) * 100), 2),
                "avg_demand_size": round(avg_demand, 4),
                "avg_interval": round(avg_interval, 2)
            },
            "smoothed_values": {
                "final_z": round(float(final_z), 4),
                "final_p": round(float(final_p), 4),
                "demand_rate": round(float(croston_forecast_value), 4)
            },
            "metrics": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 2) if mape else None
            },
            "historical": historical,
            "forecast": forecast_data
        }
        
    except Exception as e:
        print(f"CROSTON ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")

# STL Decomposition (Seasonal-Trend using LOESS)
class STLDecompositionRequest(BaseModel):
    data: List[Dict[str, Any]]
    value_column: str
    seasonal_period: int = 12

@app.post("/stl-decomposition")
def stl_decomposition(request: STLDecompositionRequest):
    from statsmodels.tsa.seasonal import STL
    
    try:
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The dataset is empty.")
        
        val_col = request.value_column
        if val_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{val_col}' not found.")
        
        # Extract values
        values = pd.to_numeric(df[val_col], errors='coerce').dropna()
        
        min_samples = 2 * request.seasonal_period + 1
        if len(values) < min_samples:
            raise HTTPException(status_code=400, detail=f"Need at least {min_samples} data points for STL with period {request.seasonal_period}.")
        
        # Reset index
        values = values.reset_index(drop=True)
        
        # Perform STL decomposition
        stl = STL(values, period=request.seasonal_period, robust=True)
        result = stl.fit()
        
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
        # Calculate strength metrics
        # Trend strength: 1 - Var(residual) / Var(trend + residual)
        var_resid = float(residual.var())
        var_trend_resid = float((trend + residual).var())
        trend_strength = max(0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0
        
        # Seasonal strength: 1 - Var(residual) / Var(seasonal + residual)
        var_seas_resid = float((seasonal + residual).var())
        seasonal_strength = max(0, 1 - var_resid / var_seas_resid) if var_seas_resid > 0 else 0
        
        # Prepare data for charts
        decomposition_data = []
        for i in range(len(values)):
            decomposition_data.append({
                "index": int(i),
                "original": float(values.iloc[i]),
                "trend": float(trend.iloc[i]),
                "seasonal": float(seasonal.iloc[i]),
                "residual": float(residual.iloc[i])
            })
        
        # Get seasonal pattern (one cycle)
        seasonal_pattern = []
        for i in range(request.seasonal_period):
            # Average seasonal component for each position in the cycle
            pattern_vals = [decomposition_data[j]["seasonal"] for j in range(i, len(decomposition_data), request.seasonal_period)]
            seasonal_pattern.append({
                "period": int(i + 1),
                "value": float(np.mean(pattern_vals))
            })
        
        return {
            "status": "success",
            "column": val_col,
            "seasonal_period": request.seasonal_period,
            "total_observations": int(len(values)),
            "strength": {
                "trend_strength": round(float(trend_strength), 4),
                "seasonal_strength": round(float(seasonal_strength), 4)
            },
            "statistics": {
                "original_mean": round(float(values.mean()), 4),
                "original_std": round(float(values.std()), 4),
                "trend_mean": round(float(trend.mean()), 4),
                "seasonal_range": round(float(seasonal.max() - seasonal.min()), 4),
                "residual_std": round(float(residual.std()), 4)
            },
            "decomposition": decomposition_data,
            "seasonal_pattern": seasonal_pattern
        }
        
    except Exception as e:
        print(f"STL DECOMPOSITION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")