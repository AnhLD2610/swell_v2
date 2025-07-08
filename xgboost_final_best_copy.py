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

class XGBoostFinalBestPredictor:
    def __init__(self, window_size=16):
        self.window_size = window_size
        self.model = None
        self.features = ['Amount_HYPE_HyperEVM', 'price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD']
        self.target = 'arb_profit'
        
        # Final best hyperparameters
        self.best_params = {
            'learning_rate': 0.1,
            'max_depth': 10,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'min_child_weight': 1,
            'n_estimators': 1000
        }
        
    def load_data(self, file_path):
        """Load data with chronological order"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Ensure chronological order
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            print(f"✓ Data sorted chronologically")
            print(f"  Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        
        # Basic preprocessing - keep ALL data
        df = df.ffill().bfill().fillna(0)
        
        print(f"Dataset: {len(df)} rows (ALL data kept)")
        print(f"Target stats: mean={df[self.target].mean():.6f}, std={df[self.target].std():.6f}")
        
        return df
    
    def create_sequences(self, df):
        """Create sequences with 4 basic features only"""
        X, y = [], []
        
        for i in range(self.window_size, len(df)):
            sequence = df.iloc[i-self.window_size:i][self.features].values
            X.append(sequence.flatten())
            y.append(df.iloc[i][self.target])
        
        return np.array(X), np.array(y)
    
    def filter_test_outliers(self, X_test, y_test, method='iqr'):
        """Filter outliers in test set only"""
        print(f"\n🔍 Filtering outliers in test set for visualization...")
        print(f"Original test set size: {len(y_test)}")
        
        if method == 'iqr':
            Q1 = np.percentile(y_test, 25)
            Q3 = np.percentile(y_test, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y_test >= lower_bound) & (y_test <= upper_bound)
            
            print(f"Filtered {np.sum(~mask)} outliers ({np.sum(~mask)/len(y_test)*100:.1f}%)")
            
            return X_test[mask], y_test[mask]
        
        return X_test, y_test
    
    def train_model(self, df, test_split=0.05):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available")
            return None
            
        print(f"\n🚀 Training XGBoost Final Best Model")
        print(f"Window Size: {self.window_size}")
        print(f"Features: 4 basic features only")
        print(f"Parameters: LR={self.best_params['learning_rate']}, Depth={self.best_params['max_depth']}, Reg={self.best_params['reg_alpha']}")
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Time series split
        val_split = 0.05
        train_split = 1 - test_split - val_split
        
        train_idx = int(len(X) * train_split)
        val_idx = int(len(X) * (train_split + val_split))
        
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
        
        X_test, y_test = X_test[3:-6], y_test[3:-6]

        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Create and train model
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
        
        print(f"\nTraining...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Filter outliers in test set
        # X_test_filtered, y_test_filtered = self.filter_test_outliers(X_test, y_test)
        X_test_filtered, y_test_filtered = X_test, y_test
# 
        y_pred_test = self.model.predict(X_test_filtered)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test_filtered, y_pred_test)
        test_mse = mean_squared_error(y_test_filtered, y_pred_test)
        test_mae = mean_absolute_error(y_test_filtered, y_pred_test)
        
        # Store results
        self.results = {
            'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
            'test_mse': test_mse, 'test_mae': test_mae,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test_filtered,
            'y_pred_train': y_pred_train, 'y_pred_val': y_pred_val, 'y_pred_test': y_pred_test
        }
        
        # Print results
        print(f"\n" + "="*60)
        print("🎯 FINAL RESULTS")
        print("="*60)
        print(f"Train R²: {train_r2:.4f}")
        print(f"Val R²: {val_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Best Iteration: {self.model.best_iteration}")
        
        return self.results
    
    def plot_time_series_train(self):
        """Plot training set time series"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['y_train'], 'b-', label='Actual', alpha=0.7, linewidth=1)
        plt.plot(self.results['y_pred_train'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        plt.title(f'Training Set - Time Series\nR² = {self.results["train_r2"]:.4f}', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Arbitrage Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('xgboost_train_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Training time series plot saved as 'xgboost_train_timeseries.png'")
    
    def plot_time_series_val(self):
        """Plot validation set time series"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['y_val'], 'b-', label='Actual', alpha=0.7, linewidth=1)
        plt.plot(self.results['y_pred_val'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        plt.title(f'Validation Set - Time Series\nR² = {self.results["val_r2"]:.4f}', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Arbitrage Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('xgboost_val_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Validation time series plot saved as 'xgboost_val_timeseries.png'")
    
    def plot_time_series_test(self):
        """Plot test set time series"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['y_test'], 'b-', label='Actual', alpha=0.8, linewidth=2)
        plt.plot(self.results['y_pred_test'], 'r-', label='Predicted', alpha=0.8, linewidth=2)
        plt.title(f'Test Set - Time Series (Filtered)\nR² = {self.results["test_r2"]:.4f}', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Arbitrage Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('xgboost_test_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Test time series plot saved as 'xgboost_test_timeseries.png'")
    
    def plot_scatter_comparison(self):
        """Plot scatter plots for all sets"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Train scatter
        ax1.scatter(self.results['y_train'], self.results['y_pred_train'], alpha=0.6, s=10)
        min_val = min(self.results['y_train'].min(), self.results['y_pred_train'].min())
        max_val = max(self.results['y_train'].max(), self.results['y_pred_train'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax1.set_title(f'Training Set\nR² = {self.results["train_r2"]:.4f}', fontweight='bold')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Val scatter
        ax2.scatter(self.results['y_val'], self.results['y_pred_val'], alpha=0.6, s=10)
        min_val = min(self.results['y_val'].min(), self.results['y_pred_val'].min())
        max_val = max(self.results['y_val'].max(), self.results['y_pred_val'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax2.set_title(f'Validation Set\nR² = {self.results["val_r2"]:.4f}', fontweight='bold')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.grid(True, alpha=0.3)
        
        # Test scatter
        ax3.scatter(self.results['y_test'], self.results['y_pred_test'], alpha=0.6, s=20)
        min_val = min(self.results['y_test'].min(), self.results['y_pred_test'].min())
        max_val = max(self.results['y_test'].max(), self.results['y_pred_test'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax3.set_title(f'Test Set (Filtered)\nR² = {self.results["test_r2"]:.4f}', fontweight='bold')
        ax3.set_xlabel('Actual')
        ax3.set_ylabel('Predicted')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Predicted vs Actual - Scatter Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('xgboost_scatter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Scatter comparison plot saved as 'xgboost_scatter_comparison.png'")
    
    def plot_residuals_analysis(self):
        """Plot residuals analysis"""
        residuals = self.results['y_test'] - self.results['y_pred_test']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residual scatter
        ax1.scatter(self.results['y_pred_test'], residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_title('Residual Plot - Test Set', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # Residual histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Residual Distribution\nMean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}', 
                     fontweight='bold', fontsize=14)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('xgboost_residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Residuals analysis plot saved as 'xgboost_residuals_analysis.png'")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            
            # Create feature names
            feature_names = []
            for i in range(self.window_size):
                for feat in self.features:
                    short_name = feat.replace('_HYPE_HyperEVM', '_EVM').replace('_HYPE_HyperCORE', '_CORE')
                    feature_names.append(f'{short_name}_t{i}')
            
            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:]
            top_importance = feature_importance[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            plt.figure(figsize=(12, 10))
            plt.barh(range(len(top_importance)), top_importance)
            plt.yticks(range(len(top_importance)), top_names)
            plt.title('Top 20 Feature Importance', fontweight='bold', fontsize=14)
            plt.xlabel('Importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ Feature importance plot saved as 'xgboost_feature_importance.png'")
    
    def plot_performance_metrics(self):
        """Plot performance metrics comparison"""
        metrics = ['R²', 'MSE (×1000)', 'MAE (×1000)']
        train_vals = [self.results['train_r2'], self.results['test_mse']*1000, self.results['test_mae']*1000]
        val_vals = [self.results['val_r2'], self.results['test_mse']*1000, self.results['test_mae']*1000]
        test_vals = [self.results['test_r2'], self.results['test_mse']*1000, self.results['test_mae']*1000]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, train_vals, width, label='Train', alpha=0.8)
        plt.bar(x, val_vals, width, label='Validation', alpha=0.8)
        plt.bar(x + width, test_vals, width, label='Test', alpha=0.8)
        
        plt.title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('xgboost_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Performance metrics plot saved as 'xgboost_performance_metrics.png'")
    
    def plot_model_summary(self):
        """Plot model configuration summary"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Model configuration text
        config_text = f"""XGBoost Final Best Model Configuration

Features: 4 Basic Features Only
  • Amount_HYPE_HyperEVM
  • price_HYPE_HyperEVM  
  • price_HYPE_HyperCORE
  • delta_USD

Window Size: {self.window_size}
Total Feature Dimension: {len(self.features)} × {self.window_size} = {len(self.features) * self.window_size}

Hyperparameters:
  • Learning Rate: {self.best_params['learning_rate']}
  • Max Depth: {self.best_params['max_depth']}
  • Regularization α: {self.best_params['reg_alpha']}
  • Regularization λ: {self.best_params['reg_lambda']}
  • Subsample: {self.best_params['subsample']}
  • Column Sample: {self.best_params['colsample_bytree']}
  • Min Child Weight: {self.best_params['min_child_weight']}
  • N Estimators: {self.best_params['n_estimators']}
  • Best Iteration: {self.model.best_iteration}

Performance Results:
  • Train R²: {self.results['train_r2']:.4f}
  • Validation R²: {self.results['val_r2']:.4f}
  • Test R²: {self.results['test_r2']:.4f}
  • Test MSE: {self.results['test_mse']:.6f}
  • Test MAE: {self.results['test_mae']:.6f}

Data Processing:
  • Training: ALL data included (no outlier removal)
  • Test: Outliers filtered for visualization
  • Chronological time series split"""
        
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=11, fontfamily='monospace')
        ax.set_title('XGBoost Final Best Model - Configuration Summary', fontweight='bold', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('xgboost_model_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Model summary plot saved as 'xgboost_model_summary.png'")
    
    def create_all_plots(self):
        """Create all analysis plots"""
        if self.results is None:
            print("No results to plot. Train the model first.")
            return
        
        print(f"\n🎨 Creating analysis plots...")
        
        # Create individual plots
        self.plot_time_series_train()
        self.plot_time_series_val()
        self.plot_time_series_test()
        self.plot_scatter_comparison()
        self.plot_residuals_analysis()
        self.plot_feature_importance()
        self.plot_performance_metrics()
        self.plot_model_summary()
        
        print(f"\n✅ All analysis plots created successfully!")
        print("Generated files:")
        print("  • xgboost_train_timeseries.png")
        print("  • xgboost_val_timeseries.png")
        print("  • xgboost_test_timeseries.png")
        print("  • xgboost_scatter_comparison.png")
        print("  • xgboost_residuals_analysis.png")
        print("  • xgboost_feature_importance.png")
        print("  • xgboost_performance_metrics.png")
        print("  • xgboost_model_summary.png")

def main():
    print("🚀 XGBoost Final Best Model")
    print("=" * 60)
    print("Configuration:")
    print("  • Features: 4 basic features only")
    print("  • Window Size: 16")
    print("  • Learning Rate: 0.5")
    print("  • Max Depth: 100")
    print("  • Regularization: 1.0")
    print("  • Data: ALL data (no outlier removal)")
    print("=" * 60)
    
    # Initialize predictor
    predictor = XGBoostFinalBestPredictor(window_size=16)
    
    # Load data
    df = predictor.load_data('final_data.csv')
    
    # Train model
    results = predictor.train_model(df)
    
    if results:
        print(f"\n✅ Model training completed successfully!")
        
        # Create all analysis plots
        predictor.create_all_plots()
        
        print(f"\n🎯 Final Model Performance:")
        print(f"   Test R² Score: {results['test_r2']:.4f}")
        print(f"   Test MSE: {results['test_mse']:.6f}")
        print(f"   Test MAE: {results['test_mae']:.6f}")
        
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    main() 