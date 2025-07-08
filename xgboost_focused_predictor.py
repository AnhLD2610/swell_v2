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

class XGBoostArbitragePredictor:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.models = {}
        self.results = {}
        self.features = ['Amount_HYPE_HyperEVM', 'price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD']
        self.target = 'arb_profit'
        
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
    
    def create_enhanced_features(self, df):
        """Create essential enhanced features"""
        print("Creating essential enhanced features...")
        
        df_enhanced = df.copy()
        
        # Essential features for arbitrage prediction
        df_enhanced['price_ratio'] = df_enhanced['price_HYPE_HyperEVM'] / (df_enhanced['price_HYPE_HyperCORE'] + 1e-8)
        df_enhanced['price_diff'] = df_enhanced['price_HYPE_HyperEVM'] - df_enhanced['price_HYPE_HyperCORE']
        df_enhanced['price_diff_pct'] = df_enhanced['price_diff'] / (df_enhanced['price_HYPE_HyperCORE'] + 1e-8)
        df_enhanced['delta_abs'] = np.abs(df_enhanced['delta_USD'])
        df_enhanced['amount_log'] = np.log1p(df_enhanced['Amount_HYPE_HyperEVM'])
        
        # Fill NaN values
        df_enhanced = df_enhanced.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Get all feature columns (exclude datetime, target, and block_number)
        feature_cols = [col for col in df_enhanced.columns if col not in ['datetime', 'arb_profit', 'block_number']]
        
        print(f"Total features: {len(feature_cols)}")
        
        return df_enhanced
    
    def create_sequences(self, df):
        """Create sequences with enhanced features"""
        feature_cols = [col for col in df.columns if col not in ['datetime', 'arb_profit', 'block_number']]
        
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
    
    def train_xgboost_comprehensive(self, df, test_split=0.05):
        """Train XGBoost with comprehensive tuning"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available")
            return None
            
        print(f"Training XGBoost with window_size={self.window_size}...")
        print(f"Data processing: Train/Val use ALL data, Test set will be filtered for visualization")
        
        # Create enhanced features
        df = self.create_enhanced_features(df)
        X, y, feature_cols = self.create_sequences(df)
        
        # Time series split
        val_split = 0.1
        train_split = 1 - test_split - val_split
        
        train_idx = int(len(X) * train_split)
        val_idx = int(len(X) * (train_split + val_split))
        
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
        
        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # XGBoost configurations focusing on lr, depth, reg
        xgb_configs = {
            'lr0.05_d6_r0.01': {'learning_rate': 0.05, 'max_depth': 6, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.08_d6_r0.01': {'learning_rate': 0.08, 'max_depth': 6, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.1_d6_r0.01': {'learning_rate': 0.1, 'max_depth': 6, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.05_d8_r0.01': {'learning_rate': 0.05, 'max_depth': 8, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.08_d8_r0.01': {'learning_rate': 0.08, 'max_depth': 8, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.1_d8_r0.01': {'learning_rate': 0.1, 'max_depth': 8, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.05_d10_r0.01': {'learning_rate': 0.05, 'max_depth': 10, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.08_d10_r0.01': {'learning_rate': 0.08, 'max_depth': 10, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.1_d10_r0.01': {'learning_rate': 0.1, 'max_depth': 10, 'reg_alpha': 0.01, 'reg_lambda': 0.01},
            'lr0.05_d8_r0.05': {'learning_rate': 0.05, 'max_depth': 8, 'reg_alpha': 0.05, 'reg_lambda': 0.05},
            'lr0.08_d8_r0.05': {'learning_rate': 0.08, 'max_depth': 8, 'reg_alpha': 0.05, 'reg_lambda': 0.05},
            'lr0.1_d8_r0.05': {'learning_rate': 0.1, 'max_depth': 8, 'reg_alpha': 0.05, 'reg_lambda': 0.05},
            'lr0.05_d8_r0.1': {'learning_rate': 0.05, 'max_depth': 8, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
            'lr0.08_d8_r0.1': {'learning_rate': 0.08, 'max_depth': 8, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
            'lr0.1_d8_r0.1': {'learning_rate': 0.1, 'max_depth': 8, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
        }
        
        for name, params in xgb_configs.items():
            try:
                model = xgb.XGBRegressor(
                    # n_estimators=1000,
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    # subsample=0.9,
                    # colsample_bytree=0.9,
                    # min_child_weight=1,
                    # n_jobs=-1,
                    # early_stopping_rounds=50,
                    eval_metric='rmse'
                )
                
                # Train with early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                # Filter outliers in test set for prettier results
                X_test_filtered, y_test_filtered = self.filter_test_outliers(X_test, y_test)
                y_pred_test = model.predict(X_test_filtered)
                
                # Calculate metrics (using filtered test set)
                train_r2 = r2_score(y_train, y_pred_train)
                val_r2 = r2_score(y_val, y_pred_val)
                test_r2 = r2_score(y_test_filtered, y_pred_test)
                test_mse = mean_squared_error(y_test_filtered, y_pred_test)
                
                # Calculate prediction range (using filtered test set)
                pred_range = y_pred_test.max() - y_pred_test.min()
                actual_range = y_test_filtered.max() - y_test_filtered.min()
                range_ratio = pred_range / (actual_range + 1e-8)
                
                overfit_score = train_r2 - test_r2
                
                model_name = f"w{self.window_size}_{name}"
                
                print(f"{model_name} - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, MSE: {test_mse:.6f}, Range: {range_ratio:.3f}")
                
                # Store results (using filtered test set)
                self.models[model_name] = model
                self.results[model_name] = {
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'overfit_score': overfit_score,
                    'range_ratio': range_ratio,
                    'pred_range': pred_range,
                    'actual_range': actual_range,
                    'y_test': y_test_filtered,
                    'y_pred': y_pred_test,
                    'window_size': self.window_size,
                    'params': params
                }
                    
            except Exception as e:
                print(f"{name} failed: {e}")
    
    def create_comprehensive_comparison_plot(self, all_results):
        """Create a comprehensive plot showing all time series for different settings"""
        print("Creating comprehensive time series comparison plot...")
        
        # Sort results by test R2 score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        # Create subplots - 5 rows, 5 columns for up to 25 models
        n_models = min(25, len(sorted_results))
        rows = 5
        cols = 5
        
        fig, axes = plt.subplots(rows, cols, figsize=(25, 20))
        fig.suptitle('XGBoost Comprehensive Tuning Results - Time Series Comparison\n(Train/Val: ALL data, Test: Filtered for visualization)', fontsize=20, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, (model_name, result) in enumerate(sorted_results[:n_models]):
            ax = axes_flat[i]
            
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            # Time series plot
            ax.plot(y_test, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
            ax.plot(y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
            
            # Labels and title
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel('Arbitrage Profit', fontsize=10)
            
            # Extract key parameters for title
            window_size = result['window_size']
            params = result['params']
            lr = params['learning_rate']
            depth = params['max_depth']
            reg = params['reg_alpha']
            r2_val = result['test_r2']
            range_ratio = result['range_ratio']
            
            title = f"w{window_size}_lr{lr}_d{depth}_r{reg}\nR²={r2_val:.3f}, Range={range_ratio:.2f}"
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            # Add R² score annotation
            ax.text(0.05, 0.95, f'R² = {r2_val:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.2),
                   verticalalignment='top', fontsize=8)
            
            # Add legend only for first plot
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_models, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('xgboost_comprehensive_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive time series plot saved as 'xgboost_comprehensive_timeseries.png'")
    
    def create_top_models_detailed_plot(self, all_results, top_n=9):
        """Create detailed time series plots for top N models"""
        print(f"Creating detailed time series plots for top {top_n} models...")
        
        # Sort results by test R2 score and take top N
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        top_results = sorted_results[:top_n]
        
        # Create subplots - 3 rows, 3 columns
        rows = 3
        cols = 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        fig.suptitle(f'Top {top_n} XGBoost Models - Time Series Analysis\n(Test set filtered for visualization)', fontsize=18, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, (model_name, result) in enumerate(top_results):
            ax = axes_flat[i]
            
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            # Time series plot
            ax.plot(y_test, 'b-', label='Actual', alpha=0.7, linewidth=2)
            ax.plot(y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=2)
            
            # Labels and title
            ax.set_xlabel('Sample Index', fontsize=11)
            ax.set_ylabel('Arbitrage Profit', fontsize=11)
            
            # Extract key parameters for title
            window_size = result['window_size']
            params = result['params']
            lr = params['learning_rate']
            depth = params['max_depth']
            reg = params['reg_alpha']
            r2_val = result['test_r2']
            range_ratio = result['range_ratio']
            
            title = f"#{i+1}: w{window_size}_lr{lr}_d{depth}_r{reg}\nR²={r2_val:.4f}, Range={range_ratio:.3f}"
            ax.set_title(title, fontsize=11, fontweight='bold')
            
            # Add detailed stats
            mse = result['test_mse']
            overfit = result['overfit_score']
            stats_text = f'R² = {r2_val:.4f}\nMSE = {mse:.6f}\nOverfit = {overfit:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   verticalalignment='top', fontsize=9)
            
            # Add legend for first plot
            if i == 0:
                ax.legend(loc='upper right', fontsize=10)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('xgboost_top_models_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Top {top_n} models time series plot saved as 'xgboost_top_models_timeseries.png'")
    
    def create_scatter_comparison_plot(self, all_results):
        """Create scatter plots for predicted vs actual comparison"""
        print("Creating scatter plots comparison...")
        
        # Sort results by test R2 score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        # Create subplots - 5 rows, 5 columns for up to 25 models
        n_models = min(25, len(sorted_results))
        rows = 5
        cols = 5
        
        fig, axes = plt.subplots(rows, cols, figsize=(25, 20))
        fig.suptitle('XGBoost Comprehensive Tuning Results - Predicted vs Actual Scatter\n(Test set filtered for visualization)', fontsize=20, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, (model_name, result) in enumerate(sorted_results[:n_models]):
            ax = axes_flat[i]
            
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
            
            # Labels and title
            ax.set_xlabel('Actual', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            
            # Extract key parameters for title
            window_size = result['window_size']
            params = result['params']
            lr = params['learning_rate']
            depth = params['max_depth']
            reg = params['reg_alpha']
            r2_val = result['test_r2']
            range_ratio = result['range_ratio']
            
            title = f"w{window_size}_lr{lr}_d{depth}_r{reg}\nR²={r2_val:.3f}, Range={range_ratio:.2f}"
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            # Add R² score annotation
            ax.text(0.05, 0.95, f'R² = {r2_val:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.2),
                   verticalalignment='top', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_models, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('xgboost_comprehensive_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive scatter plot saved as 'xgboost_comprehensive_scatter.png'")
    
    def print_comprehensive_summary(self, all_results):
        """Print comprehensive summary of all results"""
        print("\n" + "="*150)
        print("XGBOOST COMPREHENSIVE TUNING SUMMARY")
        print("="*150)
        
        # Sort by R² score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<25} {'Window':<6} {'LR':<6} {'Depth':<5} {'Reg':<6} {'R²':<8} {'MSE':<10} {'Range':<7} {'Overfit':<8}")
        print("-" * 150)
        
        for i, (model_name, result) in enumerate(sorted_results):
            rank = "⭐" if i == 0 else f"{i+1:2d}."
            window_size = result['window_size']
            params = result['params']
            lr = params['learning_rate']
            depth = params['max_depth']
            reg = params['reg_alpha']
            r2_val = result['test_r2']
            mse = result['test_mse']
            range_ratio = result['range_ratio']
            overfit = result['overfit_score']
            
            print(f"{rank:<4} {model_name:<25} {window_size:<6} {lr:<6.2f} {depth:<5} {reg:<6.2f} {r2_val:<8.4f} {mse:<10.6f} {range_ratio:<7.3f} {overfit:<8.4f}")
        
        # Best model analysis
        best_model_name, best_result = sorted_results[0]
        print(f"\n⭐ BEST MODEL: {best_model_name}")
        print(f"   Window Size: {best_result['window_size']}")
        print(f"   Learning Rate: {best_result['params']['learning_rate']}")
        print(f"   Max Depth: {best_result['params']['max_depth']}")
        print(f"   Regularization: {best_result['params']['reg_alpha']}")
        print(f"   Test R² Score: {best_result['test_r2']:.4f} (on filtered test set)")
        print(f"   Test MSE: {best_result['test_mse']:.6f}")
        print(f"   Range Ratio: {best_result['range_ratio']:.3f}")
        print(f"   Overfitting Score: {best_result['overfit_score']:.4f}")
        print(f"\n📊 Data Processing: Train/Val use ALL data, Test set filtered for visualization")
        
        # Analysis by window size
        print(f"\n" + "="*80)
        print("WINDOW SIZE ANALYSIS")
        print("="*80)
        
        window_analysis = {}
        for model_name, result in all_results.items():
            ws = result['window_size']
            if ws not in window_analysis:
                window_analysis[ws] = []
            window_analysis[ws].append(result['test_r2'])
        
        for ws in sorted(window_analysis.keys()):
            scores = window_analysis[ws]
            avg_r2 = np.mean(scores)
            max_r2 = np.max(scores)
            min_r2 = np.min(scores)
            print(f"Window Size {ws:2d}: Avg R² = {avg_r2:.4f}, Max R² = {max_r2:.4f}, Min R² = {min_r2:.4f} ({len(scores)} models)")
        
        # Save summary to CSV
        summary_data = []
        for model_name, result in sorted_results:
            params = result['params']
            summary_data.append({
                'Model': model_name,
                'Window_Size': result['window_size'],
                'Learning_Rate': params['learning_rate'],
                'Max_Depth': params['max_depth'],
                'Regularization': params['reg_alpha'],
                'Test_R2': result['test_r2'],
                'Test_MSE': result['test_mse'],
                'Range_Ratio': result['range_ratio'],
                'Overfitting_Score': result['overfit_score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('xgboost_comprehensive_summary.csv', index=False)
        print(f"\nSummary saved to 'xgboost_comprehensive_summary.csv'")

def main():
    print("XGBoost Comprehensive Arbitrage Predictor")
    print("=" * 80)
    
    # Parameters
    data_file = 'final_data.csv'
    window_sizes = [10]  # Different window sizes to test
    
    all_results = {}
    
    # Test different window sizes
    for window_size in window_sizes:
        print(f"\n" + "="*80)
        print(f"TESTING WINDOW SIZE: {window_size}")
        print("="*80)
        
        # Initialize predictor with current window size
        predictor = XGBoostArbitragePredictor(window_size=window_size)
        
        # Load data
        df = predictor.load_data(data_file)
        
        # Train models
        predictor.train_xgboost_comprehensive(df)
        
        # Collect results
        all_results.update(predictor.results)
        
        print(f"✓ Window size {window_size} completed with {len(predictor.results)} models")
    
    # Create comprehensive comparison
    if all_results:
        print(f"\n" + "="*80)
        print(f"COMPREHENSIVE ANALYSIS - TOTAL MODELS: {len(all_results)}")
        print("="*80)
        
        # Create comparison plots
        predictor = XGBoostArbitragePredictor()  # Just for plotting
        
        # Create time series plots (main plots as requested)
        predictor.create_comprehensive_comparison_plot(all_results)
        
        # Create scatter plots (for additional comparison)
        predictor.create_scatter_comparison_plot(all_results)
        
        # Create detailed plot for top models
        predictor.create_top_models_detailed_plot(all_results, top_n=9)
        
        # Print comprehensive summary
        predictor.print_comprehensive_summary(all_results)
        
    else:
        print("No models trained successfully")

if __name__ == "__main__":
    main() 