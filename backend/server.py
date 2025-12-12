from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List, Dict, Any
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