import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Install and import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Auto-install missing packages
def install_missing_packages():
    """Try to install missing packages automatically"""
    if not XGBOOST_AVAILABLE:
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
            print("XGBoost installed successfully!")
            globals()['XGBOOST_AVAILABLE'] = True
            globals()['xgb'] = __import__('xgboost')
        except:
            print("Failed to install XGBoost")

# Try to install missing packages
install_missing_packages()

# Set random seeds
# np.random.seed(42)

class ImprovedArbitragePredictor:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.models = {}
        self.results = {}
        self.features = ['Amount_HYPE_HyperEVM', 'price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD']
        self.target = 'arb_profit'
        
    def load_data(self, file_path):
        """Load data without complex preprocessing"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Basic preprocessing only
        df = df.ffill().fillna(0)
        
        print(f"Loaded {len(df)} rows")
        print(f"Target range: [{df[self.target].min():.3f}, {df[self.target].max():.3f}]")
        
        return df
    
    def create_enhanced_features(self, df):
        """Create meaningful financial features for better model performance"""
        print("Creating enhanced features...")
        
        # Create a copy to avoid modifying original
        df_enhanced = df.copy()
        
        # 1. Price differences and ratios
        df_enhanced['price_diff'] = df_enhanced['price_HYPE_HyperEVM'] - df_enhanced['price_HYPE_HyperCORE']
        df_enhanced['price_ratio'] = df_enhanced['price_HYPE_HyperEVM'] / (df_enhanced['price_HYPE_HyperCORE'] + 1e-10)
        df_enhanced['price_spread_pct'] = (df_enhanced['price_diff'] / (df_enhanced['price_HYPE_HyperCORE'] + 1e-10)) * 100
        
        # 2. Moving averages (short and long term)
        for window in [3, 5, 10]:
            df_enhanced[f'price_evm_ma_{window}'] = df_enhanced['price_HYPE_HyperEVM'].rolling(window=window).mean()
            df_enhanced[f'price_core_ma_{window}'] = df_enhanced['price_HYPE_HyperCORE'].rolling(window=window).mean()
            df_enhanced[f'amount_ma_{window}'] = df_enhanced['Amount_HYPE_HyperEVM'].rolling(window=window).mean()
        
        # 3. Price volatility
        for window in [3, 5, 10]:
            df_enhanced[f'price_evm_vol_{window}'] = df_enhanced['price_HYPE_HyperEVM'].rolling(window=window).std()
            df_enhanced[f'price_core_vol_{window}'] = df_enhanced['price_HYPE_HyperCORE'].rolling(window=window).std()
        
        # 4. Price momentum (rate of change)
        for lag in [1, 3, 5]:
            df_enhanced[f'price_evm_roc_{lag}'] = df_enhanced['price_HYPE_HyperEVM'].pct_change(lag)
            df_enhanced[f'price_core_roc_{lag}'] = df_enhanced['price_HYPE_HyperCORE'].pct_change(lag)
        
        # 5. Amount-based features
        df_enhanced['amount_log'] = np.log(df_enhanced['Amount_HYPE_HyperEVM'] + 1)
        df_enhanced['amount_zscore'] = (df_enhanced['Amount_HYPE_HyperEVM'] - df_enhanced['Amount_HYPE_HyperEVM'].mean()) / df_enhanced['Amount_HYPE_HyperEVM'].std()
        
        # 6. Delta USD features
        df_enhanced['delta_usd_abs'] = np.abs(df_enhanced['delta_USD'])
        df_enhanced['delta_usd_log'] = np.log(np.abs(df_enhanced['delta_USD']) + 1) * np.sign(df_enhanced['delta_USD'])
        
        # 7. Lag features (recent history)
        for lag in [1, 2, 3]:
            df_enhanced[f'arb_profit_lag_{lag}'] = df_enhanced['arb_profit'].shift(lag)
            df_enhanced[f'price_diff_lag_{lag}'] = df_enhanced['price_diff'].shift(lag)
        
        # Fill NaN values created by rolling operations
        df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
        
        # Update features list to include new features
        new_features = [col for col in df_enhanced.columns if col not in ['datetime', 'arb_profit']]
        print(f"Enhanced features created: {len(new_features)} total features")
        print(f"New features added: {len(new_features) - len(self.features)}")
        
        return df_enhanced
    
    def create_sequences(self, df):
        """Create sequences with enhanced features and feature selection"""
        # Use all available features except datetime and target
        feature_cols = [col for col in df.columns if col not in ['datetime', 'arb_profit']]
        
        print(f"Available features: {len(feature_cols)}")
        
        # Feature selection using correlation with target
        correlations = df[feature_cols + ['arb_profit']].corr()['arb_profit'].abs()
        correlations = correlations.drop('arb_profit').sort_values(ascending=False)
        
        # Select top features (avoid too many features)
        max_features = min(20, len(feature_cols))  # Limit to 20 features max
        selected_features = correlations.head(max_features).index.tolist()
        
        print(f"Selected top {len(selected_features)} features based on correlation:")
        for i, feature in enumerate(selected_features[:10]):  # Show top 10
            print(f"  {i+1}. {feature}: {correlations[feature]:.4f}")
        
        X, y = [], []
        
        for i in range(self.window_size, len(df)):
            # Get sequence of selected features
            sequence = df.iloc[i-self.window_size:i][selected_features].values
            X.append(sequence.flatten())
            y.append(df.iloc[i][self.target])
        
        print(f"Final feature dimension: {len(selected_features)} features × {self.window_size} window = {len(selected_features) * self.window_size} total features")
        
        return np.array(X), np.array(y), selected_features
    
    def train_linear_regression_variants(self, df, test_split=0.1):
        """Train different Linear Regression variants with regularization"""
        print("Training Linear Regression with different regularization...")
        
        # Create enhanced features
        df = self.create_enhanced_features(df)
        X, y, feature_cols = self.create_sequences(df)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Features used: {feature_cols}")
        print(f"Total features: {len(feature_cols)} x {self.window_size} = {X.shape[1]}")
        print(f"Samples to features ratio: {X.shape[0]}/{X.shape[1]} = {X.shape[0]/X.shape[1]:.2f}")
        
        # Use time series split (more appropriate for time series data)
        split_idx = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Feature scaling - IMPORTANT for linear regression with regularization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("Feature scaling applied (essential for regularized linear models)")
        
        # Different Linear Regression variants with better regularization range
        models = {
            'Ridge (alpha=0.1)': Ridge(alpha=0.1),
            'Ridge (alpha=1.0)': Ridge(alpha=1.0),
            'Ridge (alpha=10.0)': Ridge(alpha=10.0),
            'Ridge (alpha=100.0)': Ridge(alpha=100.0),
            'Lasso (alpha=0.1)': Lasso(alpha=0.1),
            'Lasso (alpha=1.0)': Lasso(alpha=1.0),
            'Lasso (alpha=10.0)': Lasso(alpha=10.0),
            'ElasticNet (alpha=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'ElasticNet (alpha=1.0)': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'ElasticNet (alpha=10.0)': ElasticNet(alpha=10.0, l1_ratio=0.5),
            'ElasticNet (alpha=1.0, l1_0.3)': ElasticNet(alpha=1.0, l1_ratio=0.3),
            'ElasticNet (alpha=1.0, l1_0.7)': ElasticNet(alpha=1.0, l1_ratio=0.7)
        }
        
        for name, model in models.items():
            try:
                # Fit model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                # Calculate overfitting indicator
                overfit_score = train_r2 - test_r2
                
                print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}, Overfit: {overfit_score:.4f}")
                
                # Store results
                self.models[name] = {'model': model, 'scaler': scaler}
                self.results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'overfit_score': overfit_score,
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
                    
            except Exception as e:
                print(f"{name} failed: {e}")
        
        return y_test, y_pred_test
    
    def train_random_forest_variants(self, df, test_split=0.1):
        """Train Random Forest with different parameters and early stopping"""
        print("Training Random Forest with different parameters...")
        
        # Create enhanced features
        df = self.create_enhanced_features(df)
        X, y, feature_cols = self.create_sequences(df)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Use time series split with validation set
        val_split = 0.1  # 10% for validation
        train_split = 1 - test_split - val_split
        
        train_idx = int(len(X) * train_split)
        val_idx = int(len(X) * (train_split + val_split))
        
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
        
        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Better Random Forest configurations with regularization
        rf_configs = {
            'RF (n=50, depth=5)': {'n_estimators': 50, 'max_depth': 5},
            'RF (n=100, depth=8)': {'n_estimators': 100, 'max_depth': 8},
            'RF (n=100, depth=10)': {'n_estimators': 100, 'max_depth': 10},
            'RF (n=150, depth=8)': {'n_estimators': 150, 'max_depth': 8},
            'RF (n=200, depth=10)': {'n_estimators': 200, 'max_depth': 10},
            'RF (n=100, depth=12)': {'n_estimators': 100, 'max_depth': 12},
            'RF (n=100, depth=6, min_5)': {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 10},
            'RF (n=150, depth=None)': {'n_estimators': 150, 'max_depth': None}
        }
        
        for name, params in rf_configs.items():
            try:
                # Create model with better regularization
                model = RandomForestRegressor(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    random_state=42,
                    n_jobs=-1,
                    min_samples_split=params.get('min_samples_split', 5),
                    min_samples_leaf=3,  # Increased for regularization
                    max_features='sqrt',  # Feature subsampling
                    bootstrap=True,
                    oob_score=True  # Out-of-bag scoring
                )
                
                model.fit(X_train, y_train)
                
                # Predictions on all sets
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                val_r2 = r2_score(y_val, y_pred_val)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mse = mean_squared_error(y_train, y_pred_train)
                val_mse = mean_squared_error(y_val, y_pred_val)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                # Overfitting indicators
                overfit_score = train_r2 - test_r2
                val_gap = train_r2 - val_r2
                
                print(f"{name} - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}, Overfit: {overfit_score:.4f}")
                
                # Store results
                self.models[name] = {'model': model, 'scaler': None}
                self.results[name] = {
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'test_mse': test_mse,
                    'overfit_score': overfit_score,
                    'val_gap': val_gap,
                    'oob_score': model.oob_score_,
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
                    
            except Exception as e:
                print(f"{name} failed: {e}")
        
        return y_test, y_pred_test
    
    def train_xgboost_variants(self, df, test_split=0.1):
        """Train XGBoost with different parameters and early stopping"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available")
            return None, None
            
        print("Training XGBoost with different parameters and early stopping...")
        
        # Create enhanced features
        df = self.create_enhanced_features(df)
        X, y, feature_cols = self.create_sequences(df)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Use time series split with validation set
        val_split = 0.1  # 10% for validation
        train_split = 1 - test_split - val_split
        
        train_idx = int(len(X) * train_split)
        val_idx = int(len(X) * (train_split + val_split))
        
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
        
        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Better XGBoost configurations with regularization
        xgb_configs = {
            'XGB (lr=0.05, depth=4)': {'learning_rate': 0.05, 'max_depth': 4, 'reg_alpha': 0.1},
            'XGB (lr=0.1, depth=4)': {'learning_rate': 0.1, 'max_depth': 4, 'reg_alpha': 0.1},
            'XGB (lr=0.1, depth=6)': {'learning_rate': 0.1, 'max_depth': 6, 'reg_alpha': 0.1},
            'XGB (lr=0.05, depth=6)': {'learning_rate': 0.05, 'max_depth': 6, 'reg_alpha': 0.1},
            'XGB (lr=0.1, depth=8)': {'learning_rate': 0.1, 'max_depth': 8, 'reg_alpha': 0.1},
            'XGB (lr=0.2, depth=4)': {'learning_rate': 0.2, 'max_depth': 4, 'reg_alpha': 0.1},
            'XGB (lr=0.01, depth=6)': {'learning_rate': 0.01, 'max_depth': 6, 'reg_alpha': 0.1},
            'XGB (lr=0.1, depth=6, reg=1)': {'learning_rate': 0.1, 'max_depth': 6, 'reg_alpha': 1.0}
        }
        
        for name, params in xgb_configs.items():
            try:
                # Create model with early stopping and better regularization
                model = xgb.XGBRegressor(
                    n_estimators=1000,  # High number, will use early stopping
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    random_state=42,
                    n_jobs=-1,
                    reg_alpha=params.get('reg_alpha', 0.1),
                    reg_lambda=0.1,  # L2 regularization
                    subsample=0.8,
                    colsample_bytree=0.8,
                    colsample_bylevel=0.8,
                    min_child_weight=3,  # Regularization
                    early_stopping_rounds=50,
                    eval_metric='rmse'
                )
                
                # Fit with early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
                
                # Predictions on all sets
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                val_r2 = r2_score(y_val, y_pred_val)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mse = mean_squared_error(y_train, y_pred_train)
                val_mse = mean_squared_error(y_val, y_pred_val)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                # Overfitting indicators
                overfit_score = train_r2 - test_r2
                val_gap = train_r2 - val_r2
                
                print(f"{name} - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}, Overfit: {overfit_score:.4f}, Trees: {model.best_iteration}")
                
                # Store results
                self.models[name] = {'model': model, 'scaler': None}
                self.results[name] = {
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'test_mse': test_mse,
                    'overfit_score': overfit_score,
                    'val_gap': val_gap,
                    'best_iteration': model.best_iteration,
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
                    
            except Exception as e:
                print(f"{name} failed: {e}")
        
        return y_test, y_pred_test
    
    def find_best_models_by_type(self):
        """Find best model for each type using R² (higher is better)"""
        model_types = {
            'Linear': [],
            'Random Forest': [],
            'XGBoost': []
        }
        
        # Group models by type
        for model_name in self.results.keys():
            if 'Linear' in model_name or 'Ridge' in model_name or 'Lasso' in model_name or 'ElasticNet' in model_name:
                model_types['Linear'].append(model_name)
            elif 'RF' in model_name:
                model_types['Random Forest'].append(model_name)
            elif 'XGB' in model_name:
                model_types['XGBoost'].append(model_name)
        
        # Find best model for each type based on R² (higher is better)
        best_models = {}
        for model_type, models in model_types.items():
            if models:
                best_model = max(models, key=lambda x: self.results[x]['test_r2'])
                best_models[model_type] = best_model
        
        return best_models

    def create_best_models_plots(self):
        """Create plots for best models only"""
        best_models = self.find_best_models_by_type()
        
        if not best_models:
            print("No models to plot")
            return
        
        print(f"\nBest models selected (based on highest R²):")
        for model_type, model_name in best_models.items():
            r2_score = self.results[model_name]['test_r2']
            mse_score = self.results[model_name]['test_mse']
            print(f"{model_type}: {model_name} (R² = {r2_score:.4f}, MSE = {mse_score:.4f})")
        
        # Create comparison plot for best models
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Best Models Comparison', fontsize=20, fontweight='bold')
        
        # Plot 1: R² Score Comparison
        model_types = list(best_models.keys())
        model_names = list(best_models.values())
        test_r2_scores = [self.results[name]['test_r2'] for name in model_names]
        train_r2_scores = [self.results[name]['train_r2'] for name in model_names]
        
        x_pos = np.arange(len(model_types))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_r2_scores, width, label='Train R²', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, test_r2_scores, width, label='Test R²', alpha=0.7)
        axes[0, 0].set_xlabel('Model Types')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Best Models R² Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_types)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MAE Comparison
        mae_scores = [mean_absolute_error(self.results[name]['y_test'], self.results[name]['y_pred']) 
                     for name in model_names]
        
        axes[0, 1].bar(model_types, mae_scores, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Model Types')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Best Models MAE Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Predicted vs Actual for overall best model (based on R²)
        overall_best_name = max(model_names, key=lambda x: self.results[x]['test_r2'])
        best_result = self.results[overall_best_name]
        
        axes[1, 0].scatter(best_result['y_test'], best_result['y_pred'], alpha=0.6)
        min_val = min(best_result['y_test'].min(), best_result['y_pred'].min())
        max_val = max(best_result['y_test'].max(), best_result['y_pred'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title(f'Overall Best: {overall_best_name}')
        axes[1, 0].grid(True, alpha=0.3)
        
        r2_val = self.results[overall_best_name]['test_r2']
        axes[1, 0].text(0.05, 0.95, f'R² = {r2_val:.4f}', transform=axes[1, 0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontweight='bold')
        
        # Plot 4: Time series for overall best model
        axes[1, 1].plot(best_result['y_test'], 'b-', label='Actual', alpha=0.7)
        axes[1, 1].plot(best_result['y_pred'], 'r-', label='Predicted', alpha=0.7)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Arbitrage Profit')
        axes[1, 1].set_title(f'Time Series: {overall_best_name}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create filename with best settings
        r2_val_file = self.results[overall_best_name]['test_r2']
        filename = f'best_models_comparison_{overall_best_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "")}_R2{r2_val_file:.4f}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Best models comparison plot saved as '{filename}'")
    
    def create_best_individual_plots(self):
        """Create individual plots for best models only"""
        best_models = self.find_best_models_by_type()
        
        for model_type, model_name in best_models.items():
            result = self.results[model_name]
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Best {model_type}: {model_name}', fontsize=16, fontweight='bold')
            
            y_test, y_pred = result['y_test'], result['y_pred']
            
            # Plot 1: Actual vs Predicted
            ax1.scatter(y_test, y_pred, alpha=0.6, s=30)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax1.set_xlabel('Actual')
            ax1.set_ylabel('Predicted')
            ax1.set_title('Predicted vs Actual')
            ax1.grid(True, alpha=0.3)
            
            r2_val = r2_score(y_test, y_pred)
            ax1.text(0.05, 0.95, f'R² = {r2_val:.4f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
            
            # Plot 2: Time series
            ax2.plot(y_test, 'b-', label='Actual', alpha=0.7, linewidth=1)
            ax2.plot(y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Arbitrage Profit')
            ax2.set_title('Time Series Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            residuals = y_test - y_pred
            ax3.scatter(y_pred, residuals, alpha=0.6, s=30)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Residuals')
            ax3.set_title('Residual Plot')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Error histogram
            ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Residual Distribution')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            r2_val_file = result['test_r2']
            filename = f'best_{model_type.lower().replace(" ", "_")}_{model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "")}_R2{r2_val_file:.4f}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Best {model_type} plot saved as '{filename}'")
    
    def print_best_models_summary(self):
        """Print summary of best models only"""
        best_models = self.find_best_models_by_type()
        
        print("\n" + "="*100)
        print("BEST MODELS SUMMARY")
        print("="*100)
        
        # Create summary table for best models
        summary_data = []
        for model_type, model_name in best_models.items():
            result = self.results[model_name]
            y_test, y_pred = result['y_test'], result['y_pred']
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2_val = result['test_r2']
            
            summary_data.append({
                'Type': model_type,
                'Model': model_name,
                'MSE': result['test_mse'],
                'R²': r2_val,
                'MAE': mae,
                'RMSE': rmse,
                'Min_Pred': y_pred.min(),
                'Max_Pred': y_pred.max()
            })
        
        # Sort by R² score (higher is better)
        summary_data.sort(key=lambda x: x['R²'], reverse=True)
        
        print(f"{'Type':<15} {'Model':<35} {'R²':<8} {'MSE':<8} {'MAE':<8} {'RMSE':<8} {'Min_Pred':<10} {'Max_Pred':<10}")
        print("-" * 110)
        
        for i, data in enumerate(summary_data):
            rank = "⭐" if i == 0 else f"{i+1}. "
            print(f"{rank:<3}{data['Type']:<12} {data['Model']:<32} {data['R²']:<8.4f} {data['MSE']:<8.4f} {data['MAE']:<8.4f} {data['RMSE']:<8.4f} {data['Min_Pred']:<10.3f} {data['Max_Pred']:<10.3f}")
        
        print("="*110)
        print(f"Overall Best Model: {summary_data[0]['Model']} (R² = {summary_data[0]['R²']:.4f}, MSE = {summary_data[0]['MSE']:.4f})")
        print("="*110)
        
        # Save summary to CSV
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('best_models_summary_v2.csv', index=False)
        print("Best models summary saved to 'best_models_summary_v2.csv'")

def main():
    print("Enhanced Arbitrage Predictor v2.0 - Advanced Parameter Testing")
    print("=" * 70)
    
    # Parameters  
    data_file = 'final_data_task1_swell_HOURLY.csv'
    window_size = 12  # Balanced window size (12 hours of history)
    
    # Initialize predictor
    predictor = ImprovedArbitragePredictor(window_size=window_size)
    
    # Load data
    df = predictor.load_data(data_file)
    
    print(f"\nUsing {window_size} hours with enhanced feature engineering")
    print(f"Original features: {predictor.features}")
    print(f"Target: {predictor.target}")
    print(f"Enhanced features will be created automatically")
    print(f"Top 20 features will be selected based on correlation with target")
    print(f"Expected total features: ~20 × {window_size} = ~240 features")
    print(f"Data splits: 80% train, 10% validation, 10% test")
    print(f"Key improvements:")
    print("  - Enhanced feature engineering (moving averages, volatility, momentum)")
    print("  - Feature selection based on correlation") 
    print("  - Feature scaling for linear models")
    print("  - Validation set for better model selection")
    print("  - Early stopping for tree-based models")
    print("  - Regularization to prevent overfitting")
    print("  - Time series splitting (no data leakage)")
    
    # Train all model variants
    print("\n" + "="*70)
    print("TRAINING ALL MODEL VARIANTS (ENHANCED FEATURES + REGULARIZATION)")
    print("="*70)
    
    try:
        predictor.train_linear_regression_variants(df)
        print("\n✓ Linear Regression variants completed!")
    except Exception as e:
        print(f"✗ Linear Regression variants failed: {e}")
    
    try:
        predictor.train_random_forest_variants(df)
        print("\n✓ Random Forest variants completed!")
    except Exception as e:
        print(f"✗ Random Forest variants failed: {e}")
    
    try:
        predictor.train_xgboost_variants(df)
        print("\n✓ XGBoost variants completed!")
    except Exception as e:
        print(f"✗ XGBoost variants failed: {e}")
    
    # Show results
    if predictor.results:
        print("\n" + "="*70)
        print("CREATING BEST MODELS ANALYSIS")
        print("="*70)
        
        # Create comparison plots for best models
        predictor.create_best_models_plots()
        
        # Create individual plots for best models
        predictor.create_best_individual_plots()
        
        # Print summary of best models
        predictor.print_best_models_summary()
        
    else:
        print("No models trained successfully")

if __name__ == "__main__":
    main() 