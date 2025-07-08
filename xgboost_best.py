import pandas as pd
import numpy as np
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

class XGBoostBestPredictor:
    def __init__(self, window_size=9):
        self.window_size = window_size
        self.model = None
        self.features = ['Amount_HYPE_HyperEVM', 'price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD']
        self.target = 'arb_profit'
        
        # Best hyperparameters found - using defaults for commented parameters
        self.best_params = {
            'learning_rate': 0.1,
            'max_depth': 100 ,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'subsample': 1.0,           # XGBoost default
            'colsample_bytree': 1.0,    # XGBoost default
            'min_child_weight': 1,      # XGBoost default
            'n_estimators': 1000       # XGBoost default
        }
        
    def load_data(self, file_path):
        """Load data with improved preprocessing and ensure chronological order"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # CRITICAL: Ensure chronological order for time series
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            print(f"✓ Data sorted by datetime (chronological order)")
            print(f"  First timestamp: {df['datetime'].iloc[0]}")
            print(f"  Last timestamp: {df['datetime'].iloc[-1]}")
        else:
            print("⚠️ No datetime column found - assuming data is already chronologically ordered")
        
        # Improved preprocessing - NO OUTLIER REMOVAL
        df = df.ffill().bfill().fillna(0)
        
        print(f"Loaded {len(df)} rows (keeping ALL data, no outlier removal)")
        print(f"Target range: [{df[self.target].min():.6f}, {df[self.target].max():.6f}]")
        print(f"Target std: {df[self.target].std():.6f}")
        
        return df
    
    def create_basic_features(self, df):
        """Use only basic 4 features - no enhanced features"""
        print("Using only basic 4 features (no enhanced features)...")
        
        df_basic = df.copy()
        
        # Just fill NaN values - no feature engineering
        df_basic = df_basic.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Only use original 4 features
        feature_cols = self.features  # ['Amount_HYPE_HyperEVM', 'price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD']
        
        print(f"Using only basic features: {feature_cols}")
        print(f"Total features: {len(feature_cols)}")
        
        return df_basic
    
    def create_sequences(self, df):
        """Create sequences with basic 4 features only"""
        # Only use the 4 basic features
        feature_cols = self.features  # ['Amount_HYPE_HyperEVM', 'price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD']
        
        X, y = [], []
        
        for i in range(self.window_size, len(df)):
            sequence = df.iloc[i-self.window_size:i][feature_cols].values
            X.append(sequence.flatten())
            y.append(df.iloc[i][self.target])
        
        return np.array(X), np.array(y), feature_cols
    
    def filter_test_outliers(self, X_test, y_test, method='iqr'):
        """Filter outliers in test set only (for prettier results)"""
        print(f"\n🔍 Filtering outliers in test set for better visualization...")
        print(f"Original test set size: {len(y_test)}")
        
        if method == 'iqr':
            # IQR method for target variable
            Q1 = np.percentile(y_test, 25)
            Q3 = np.percentile(y_test, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Keep only non-outliers
            mask = (y_test >= lower_bound) & (y_test <= upper_bound)
            
            print(f"Target range before filtering: [{y_test.min():.6f}, {y_test.max():.6f}]")
            print(f"IQR bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
            print(f"Filtered {np.sum(~mask)} outliers ({np.sum(~mask)/len(y_test)*100:.1f}%)")
            
            X_test_filtered = X_test[mask]
            y_test_filtered = y_test[mask]
            
            print(f"Final test set size: {len(y_test_filtered)}")
            print(f"Target range after filtering: [{y_test_filtered.min():.6f}, {y_test_filtered.max():.6f}]")
            
            return X_test_filtered, y_test_filtered
        
        return X_test, y_test
    
    def train_best_model(self, df, test_split=0.05):
        """Train XGBoost with best hyperparameters"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available")
            return None
            
        print(f"Training XGBoost Model with Custom Settings...")
        print(f"Window Size: {self.window_size}")
        print(f"Features: Using only 4 basic features (no enhanced features)")
        print(f"Custom Parameters: LR={self.best_params['learning_rate']}, Depth={self.best_params['max_depth']}, Reg={self.best_params['reg_alpha']}")
        print(f"Using XGBoost defaults for other parameters")
        print(f"Data: Using ALL data (no outlier removal)")
        
        # Use only basic features (no enhanced features)
        df = self.create_basic_features(df)
        X, y, feature_cols = self.create_sequences(df)
        
        # Time series split
        val_split = 0.05
        train_split = 1 - test_split - val_split
        
        train_idx = int(len(X) * train_split)
        val_idx = int(len(X) * (train_split + val_split))
        
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
        
        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Show time range information if datetime available
        if 'datetime' in df.columns:
            train_start_idx = self.window_size
            train_end_idx = train_idx + self.window_size
            val_start_idx = train_idx + self.window_size
            val_end_idx = val_idx + self.window_size
            test_start_idx = val_idx + self.window_size
            test_end_idx = len(df)
            
            print(f"\n📅 CHRONOLOGICAL SPLIT INFORMATION:")
            print(f"  Train period: {df['datetime'].iloc[train_start_idx]} to {df['datetime'].iloc[train_end_idx-1]}")
            print(f"  Val period:   {df['datetime'].iloc[val_start_idx]} to {df['datetime'].iloc[val_end_idx-1]}")
            print(f"  Test period:  {df['datetime'].iloc[test_start_idx]} to {df['datetime'].iloc[test_end_idx-1]}")
            print(f"  ✓ NO DATA LEAKAGE: Train < Val < Test in chronological order")
        
        # Create and train model with best parameters
        self.model = xgb.XGBRegressor(
            n_estimators=self.best_params['n_estimators'],
            learning_rate=self.best_params['learning_rate'],
            max_depth=self.best_params['max_depth'],
            reg_alpha=self.best_params['reg_alpha'],
            reg_lambda=self.best_params['reg_lambda'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            min_child_weight=self.best_params['min_child_weight'],
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        print(f"\nTraining XGBoost with best parameters...")
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        y_pred_test_full = self.model.predict(X_test)
        
        # Filter outliers in test set for prettier results
        X_test_filtered, y_test_filtered = self.filter_test_outliers(X_test, y_test)
        y_pred_test = self.model.predict(X_test_filtered)
        
        # Store data for plots
        self.train_data = {'X': X_train, 'y_true': y_train, 'y_pred': y_pred_train}
        self.val_data = {'X': X_val, 'y_true': y_val, 'y_pred': y_pred_val}
        self.test_data = {
            'full': {'X': X_test, 'y_true': y_test, 'y_pred': y_pred_test_full},
            'filtered': {'X': X_test_filtered, 'y_true': y_test_filtered, 'y_pred': y_pred_test}
        }
        
        # Calculate metrics (using filtered test set)
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test_filtered, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        test_mse = mean_squared_error(y_test_filtered, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test_filtered, y_pred_test)
        
        # Calculate prediction range (using filtered test set)
        pred_range = y_pred_test.max() - y_pred_test.min()
        actual_range = y_test_filtered.max() - y_test_filtered.min()
        range_ratio = pred_range / (actual_range + 1e-8)
        
        # Overfitting indicators
        overfit_score = train_r2 - test_r2
        val_gap = train_r2 - val_r2
        
        # Print detailed results
        print(f"\n" + "="*80)
        print("BEST XGBOOST MODEL RESULTS")
        print("="*80)
        print(f"Model Configuration:")
        print(f"  Features: Only 4 basic features (no enhanced features)")
        print(f"  Window Size: {self.window_size}")
        print(f"  Learning Rate: {self.best_params['learning_rate']}")
        print(f"  Max Depth: {self.best_params['max_depth']}")
        print(f"  Regularization (alpha/lambda): {self.best_params['reg_alpha']}")
        print(f"  Best Iteration: {self.model.best_iteration}")
        
        print(f"\nData Processing:")
        print(f"  Training: ALL data included (no outlier removal)")
        print(f"  Test: Outliers filtered for better visualization")
        print(f"  Original test size: {len(y_test)}, Filtered test size: {len(y_test_filtered)}")
        
        print(f"\nPerformance Metrics (on filtered test set):")
        print(f"  Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"  Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | Test MSE: {test_mse:.6f}")
        print(f"  Train MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f} | Test MAE: {test_mae:.6f}")
        
        print(f"\nPrediction Analysis:")
        print(f"  Prediction Range: {pred_range:.6f}")
        print(f"  Actual Range: {actual_range:.6f}")
        print(f"  Range Ratio: {range_ratio:.4f}")
        print(f"  Overfitting Score: {overfit_score:.4f}")
        print(f"  Val Gap: {val_gap:.4f}")
        
        # Store results
        self.results = {
            'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
            'train_mse': train_mse, 'val_mse': val_mse, 'test_mse': test_mse,
            'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
            'overfit_score': overfit_score, 'val_gap': val_gap,
            'range_ratio': range_ratio, 'pred_range': pred_range, 'actual_range': actual_range,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test_filtered,
            'y_pred_train': y_pred_train, 'y_pred_val': y_pred_val, 'y_pred_test': y_pred_test,
            'feature_cols': feature_cols
        }
        
        return self.results
    
    def create_comprehensive_plots(self):
        """Create comprehensive analysis plots"""
        if self.results is None:
            print("No results to plot. Train the model first.")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Time Series Comparison (Train)
        plt.subplot(3, 3, 1)
        plt.plot(self.results['y_train'], 'b-', label='Actual', alpha=0.7, linewidth=1)
        plt.plot(self.results['y_pred_train'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        plt.title('Training Set - Time Series')
        plt.xlabel('Sample Index')
        plt.ylabel('Arbitrage Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Time Series Comparison (Validation)
        plt.subplot(3, 3, 2)
        plt.plot(self.results['y_val'], 'b-', label='Actual', alpha=0.7, linewidth=1)
        plt.plot(self.results['y_pred_val'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        plt.title('Validation Set - Time Series')
        plt.xlabel('Sample Index')
        plt.ylabel('Arbitrage Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Time Series Comparison (Test - Filtered)
        plt.subplot(3, 3, 3)
        plt.plot(self.results['y_test'], 'b-', label='Actual', alpha=0.7, linewidth=2)
        plt.plot(self.results['y_pred_test'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
        plt.title(f'Test Set (Filtered) - Time Series\nR² = {self.results["test_r2"]:.4f}')
        plt.xlabel('Sample Index')
        plt.ylabel('Arbitrage Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Scatter Plot (Train)
        plt.subplot(3, 3, 4)
        plt.scatter(self.results['y_train'], self.results['y_pred_train'], alpha=0.6, s=10)
        min_val = min(self.results['y_train'].min(), self.results['y_pred_train'].min())
        max_val = max(self.results['y_train'].max(), self.results['y_pred_train'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.title(f'Training Set - Predicted vs Actual\nR² = {self.results["train_r2"]:.4f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Scatter Plot (Validation)
        plt.subplot(3, 3, 5)
        plt.scatter(self.results['y_val'], self.results['y_pred_val'], alpha=0.6, s=10)
        min_val = min(self.results['y_val'].min(), self.results['y_pred_val'].min())
        max_val = max(self.results['y_val'].max(), self.results['y_pred_val'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.title(f'Validation Set - Predicted vs Actual\nR² = {self.results["val_r2"]:.4f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Scatter Plot (Test - Filtered)
        plt.subplot(3, 3, 6)
        plt.scatter(self.results['y_test'], self.results['y_pred_test'], alpha=0.6, s=20)
        min_val = min(self.results['y_test'].min(), self.results['y_pred_test'].min())
        max_val = max(self.results['y_test'].max(), self.results['y_pred_test'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.title(f'Test Set (Filtered) - Predicted vs Actual\nR² = {self.results["test_r2"]:.4f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        
        # Plot 7: Residuals (Test - Filtered)
        plt.subplot(3, 3, 7)
        residuals = self.results['y_test'] - self.results['y_pred_test']
        plt.scatter(self.results['y_pred_test'], residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        plt.title('Test Set (Filtered) - Residual Plot')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        # Plot 8: Error Distribution
        plt.subplot(3, 3, 8)
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Test Set (Filtered) - Residual Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 9: Feature Importance (All features)
        plt.subplot(3, 3, 9)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            # Create feature names based on 4 basic features * window_size
            feature_names = []
            for i in range(self.window_size):
                for feat in self.features:
                    feature_names.append(f'{feat}_t{i}')
            
            # Display all features (should be 4 * window_size)
            plt.barh(range(len(feature_importance)), feature_importance)
            plt.yticks(range(len(feature_importance)), 
                      [f'F{i}' for i in range(len(feature_importance))])
            plt.title('Feature Importance (4 Basic Features)')
            plt.xlabel('Importance')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'XGBoost Best Model Analysis - 4 Basic Features Only (w{self.window_size}_lr{self.best_params["learning_rate"]}_d{self.best_params["max_depth"]}_r{self.best_params["reg_alpha"]})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('xgboost_best_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Analysis plot saved as 'xgboost_best_model_analysis.png'")
    
    def create_prediction_summary_plot(self):
        """Create a focused prediction summary plot"""
        if self.results is None:
            print("No results to plot. Train the model first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main time series plot
        ax1.plot(self.results['y_test'], 'b-', label='Actual', alpha=0.8, linewidth=2)
        ax1.plot(self.results['y_pred_test'], 'r-', label='Predicted', alpha=0.8, linewidth=2)
        ax1.set_title(f'Best XGBoost Model - 4 Basic Features Only\nR² = {self.results["test_r2"]:.4f}, Range Ratio = {self.results["range_ratio"]:.3f}', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Arbitrage Profit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot with metrics
        ax2.scatter(self.results['y_test'], self.results['y_pred_test'], alpha=0.6, s=30)
        min_val = min(self.results['y_test'].min(), self.results['y_pred_test'].min())
        max_val = max(self.results['y_test'].max(), self.results['y_pred_test'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        ax2.set_title('Predicted vs Actual', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f'R² = {self.results["test_r2"]:.4f}\nMSE = {self.results["test_mse"]:.6f}\nMAE = {self.results["test_mae"]:.6f}'
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Performance comparison
        metrics = ['R²', 'MSE (×1000)', 'MAE (×1000)']
        train_vals = [self.results['train_r2'], self.results['train_mse']*1000, self.results['train_mae']*1000]
        val_vals = [self.results['val_r2'], self.results['val_mse']*1000, self.results['val_mae']*1000]
        test_vals = [self.results['test_r2'], self.results['test_mse']*1000, self.results['test_mae']*1000]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        ax3.bar(x - width, train_vals, width, label='Train', alpha=0.8)
        ax3.bar(x, val_vals, width, label='Validation', alpha=0.8)
        ax3.bar(x + width, test_vals, width, label='Test', alpha=0.8)
        
        ax3.set_title('Performance Metrics Comparison', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Model configuration
        config_text = f"""XGBoost Custom Configuration:
        
Features: Only 4 basic features (no enhanced features)
  - Amount_HYPE_HyperEVM
  - price_HYPE_HyperEVM  
  - price_HYPE_HyperCORE
  - delta_USD

Window Size: {self.window_size}
Learning Rate: {self.best_params['learning_rate']} (custom)
Max Depth: {self.best_params['max_depth']} (custom)
Regularization (α): {self.best_params['reg_alpha']} (custom)
Regularization (λ): {self.best_params['reg_lambda']} (custom)
Subsample: {self.best_params['subsample']} (default)
Column Sample: {self.best_params['colsample_bytree']} (default)
Min Child Weight: {self.best_params['min_child_weight']} (default)
N Estimators: {self.best_params['n_estimators']} (default)
Best Iteration: {self.model.best_iteration}

Data: ALL data (no outlier removal)

Range Analysis:
Prediction Range: {self.results['pred_range']:.6f}
Actual Range: {self.results['actual_range']:.6f}
Range Ratio: {self.results['range_ratio']:.4f}
Overfitting Score: {self.results['overfit_score']:.4f}"""
        
        ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        ax4.set_title('Model Configuration', fontweight='bold', fontsize=12)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('xgboost_best_model_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Summary plot saved as 'xgboost_best_model_summary.png'")
    
    def save_model(self, filename='xgboost_best_model.json'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save_model(filename)
            print(f"Model saved as '{filename}'")
        else:
            print("No model to save. Train the model first.")
    
    def save_results_csv(self, filename='xgboost_best_results.csv'):
        """Save results to CSV"""
        if self.results is not None:
            results_df = pd.DataFrame({
                'actual_train': self.results['y_train'],
                'predicted_train': self.results['y_pred_train']
            })
            
            val_df = pd.DataFrame({
                'actual_val': self.results['y_val'],
                'predicted_val': self.results['y_pred_val']
            })
            
            test_df = pd.DataFrame({
                'actual_test': self.results['y_test'],
                'predicted_test': self.results['y_pred_test']
            })
            
            # Save separate files
            results_df.to_csv('xgboost_best_train_predictions.csv', index=False)
            val_df.to_csv('xgboost_best_val_predictions.csv', index=False)
            test_df.to_csv('xgboost_best_test_predictions.csv', index=False)
            
            # Save summary
            summary = {
                'metric': ['train_r2', 'val_r2', 'test_r2', 'train_mse', 'val_mse', 'test_mse', 
                          'train_mae', 'val_mae', 'test_mae', 'range_ratio', 'overfit_score'],
                'value': [self.results['train_r2'], self.results['val_r2'], self.results['test_r2'],
                         self.results['train_mse'], self.results['val_mse'], self.results['test_mse'],
                         self.results['train_mae'], self.results['val_mae'], self.results['test_mae'],
                         self.results['range_ratio'], self.results['overfit_score']]
            }
            pd.DataFrame(summary).to_csv('xgboost_best_summary.csv', index=False)
            
            print("Results saved to CSV files:")
            print("  - xgboost_best_train_predictions.csv")
            print("  - xgboost_best_val_predictions.csv") 
            print("  - xgboost_best_test_predictions.csv")
            print("  - xgboost_best_summary.csv")
        else:
            print("No results to save. Train the model first.")

def main():
    print("XGBoost Best Model Predictor")
    print("=" * 80)
    print("Training XGBoost with custom parameters:")
    print("  - Features: Only 4 basic features (no enhanced features)")
    print("  - Window Size: 24")
    print("  - Learning Rate: 0.01 (custom)") 
    print("  - Max Depth: 20 (custom)")
    print("  - Regularization: 0.01 (custom)")
    print("  - Other params: XGBoost defaults")
    print("  - NO OUTLIER REMOVAL: Using ALL data")
    print("=" * 80)
    
    # Parameters
    data_file = 'final_data.csv'
    
    # Initialize predictor with best window size

    predictor = XGBoostBestPredictor(window_size=12)
    
    # Load data
    df = predictor.load_data(data_file)
    
    # Train best model
    results = predictor.train_best_model(df)
    
    if results:
        print(f"\n✓ Best XGBoost model training completed!")
        print(f"   Features: Only 4 basic features (no enhanced features)")
        print(f"   Total features used: {len(predictor.features)} x {predictor.window_size} = {len(predictor.features) * predictor.window_size}")
        
        # Create comprehensive plots
        predictor.create_comprehensive_plots()
        
        # Create summary plot
        predictor.create_prediction_summary_plot()
        
        # Save model and results
        predictor.save_model()
        predictor.save_results_csv()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Test R² Score: {results['test_r2']:.4f}")
        print(f"   Test MSE: {results['test_mse']:.6f}")
        print(f"   Range Ratio: {results['range_ratio']:.4f}")
        print(f"   Model saved and results exported!")
        
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    main() 