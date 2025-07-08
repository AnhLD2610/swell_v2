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

class XGBoostSplitFilesPredictor:
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
        
    def load_split_data(self, train_file='train_data.csv', val_file='val_data.csv', test_file='test_data.csv'):
        """Load data từ 3 file riêng biệt"""
        print(f"📖 Đọc dữ liệu từ các file riêng biệt...")
        
        # Đọc các file
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        print(f"✓ Train: {len(train_df)} rows - {train_file}")
        print(f"✓ Val:   {len(val_df)} rows - {val_file}")
        print(f"✓ Test:  {len(test_df)} rows - {test_file}")
        
        # Kiểm tra datetime nếu có
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                print(f"   {name}: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
        
        # Basic preprocessing
        train_df = train_df.ffill().bfill().fillna(0)
        val_df = val_df.ffill().bfill().fillna(0)
        test_df = test_df.ffill().bfill().fillna(0)
        
        # Thống kê target
        print(f"\n📈 Thống kê target '{self.target}':")
        print(f"   Train: mean={train_df[self.target].mean():.6f}, std={train_df[self.target].std():.6f}")
        print(f"   Val:   mean={val_df[self.target].mean():.6f}, std={val_df[self.target].std():.6f}")
        print(f"   Test:  mean={test_df[self.target].mean():.6f}, std={test_df[self.target].std():.6f}")
        
        return train_df, val_df, test_df
    
    def create_sequences(self, df):
        """Create sequences với 4 basic features"""
        X, y = [], []
        
        for i in range(self.window_size, len(df)):
            sequence = df.iloc[i-self.window_size:i][self.features].values
            X.append(sequence.flatten())
            y.append(df.iloc[i][self.target])
        
        return np.array(X), np.array(y)
    
    def train_model_from_files(self, train_file='train_data.csv', val_file='val_data.csv', test_file='test_data.csv'):
        """Train XGBoost model từ 3 file riêng biệt"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available")
            return None
            
        print(f"\n🚀 Training XGBoost từ Split Files")
        print(f"Window Size: {self.window_size}")
        print(f"Features: 4 basic features only")
        print(f"Parameters: LR={self.best_params['learning_rate']}, Depth={self.best_params['max_depth']}, Reg={self.best_params['reg_alpha']}")
        
        # Load data từ 3 file
        train_df, val_df, test_df = self.load_split_data(train_file, val_file, test_file)
        
        # Create sequences cho từng set
        print(f"\n🔄 Tạo sequences...")
        X_train, y_train = self.create_sequences(train_df)
        X_val, y_val = self.create_sequences(val_df)
        X_test, y_test = self.create_sequences(test_df)
        
        print(f"✓ Sequences created:")
        print(f"   Train: {len(X_train)} sequences")
        print(f"   Val:   {len(X_val)} sequences")
        print(f"   Test:  {len(X_test)} sequences")
        
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
        
        print(f"\n🎯 Training...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Store results
        self.results = {
            'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
            'test_mse': test_mse, 'test_mae': test_mae,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_val': y_pred_val, 'y_pred_test': y_pred_test
        }
        
        # Print results
        print(f"\n" + "="*60)
        print("🎯 FINAL RESULTS (Split Files)")
        print("="*60)
        print(f"Train R²: {train_r2:.4f}")
        print(f"Val R²: {val_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Best Iteration: {self.model.best_iteration}")
        
        return self.results
    
    def plot_all_comparison(self):
        """Plot so sánh tất cả các sets"""
        if self.results is None:
            print("No results to plot. Train the model first.")
            return
        
        # Time series plots
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Training time series
        axes[0,0].plot(self.results['y_train'], 'b-', label='Actual', alpha=0.7, linewidth=1)
        axes[0,0].plot(self.results['y_pred_train'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        axes[0,0].set_title(f'Training Set - Time Series\nR² = {self.results["train_r2"]:.4f}', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Training scatter
        axes[0,1].scatter(self.results['y_train'], self.results['y_pred_train'], alpha=0.6, s=10)
        min_val = min(self.results['y_train'].min(), self.results['y_pred_train'].min())
        max_val = max(self.results['y_train'].max(), self.results['y_pred_train'].max())
        axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0,1].set_title(f'Training Set - Scatter\nR² = {self.results["train_r2"]:.4f}', fontweight='bold')
        axes[0,1].set_xlabel('Actual')
        axes[0,1].set_ylabel('Predicted')
        axes[0,1].grid(True, alpha=0.3)
        
        # Validation time series
        axes[1,0].plot(self.results['y_val'], 'b-', label='Actual', alpha=0.7, linewidth=1)
        axes[1,0].plot(self.results['y_pred_val'], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        axes[1,0].set_title(f'Validation Set - Time Series\nR² = {self.results["val_r2"]:.4f}', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Validation scatter
        axes[1,1].scatter(self.results['y_val'], self.results['y_pred_val'], alpha=0.6, s=10)
        min_val = min(self.results['y_val'].min(), self.results['y_pred_val'].min())
        max_val = max(self.results['y_val'].max(), self.results['y_pred_val'].max())
        axes[1,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[1,1].set_title(f'Validation Set - Scatter\nR² = {self.results["val_r2"]:.4f}', fontweight='bold')
        axes[1,1].set_xlabel('Actual')
        axes[1,1].set_ylabel('Predicted')
        axes[1,1].grid(True, alpha=0.3)
        
        # Test time series
        axes[2,0].plot(self.results['y_test'], 'b-', label='Actual', alpha=0.8, linewidth=2)
        axes[2,0].plot(self.results['y_pred_test'], 'r-', label='Predicted', alpha=0.8, linewidth=2)
        axes[2,0].set_title(f'Test Set - Time Series\nR² = {self.results["test_r2"]:.4f}', fontweight='bold')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # Test scatter
        axes[2,1].scatter(self.results['y_test'], self.results['y_pred_test'], alpha=0.6, s=20)
        min_val = min(self.results['y_test'].min(), self.results['y_pred_test'].min())
        max_val = max(self.results['y_test'].max(), self.results['y_pred_test'].max())
        axes[2,1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[2,1].set_title(f'Test Set - Scatter\nR² = {self.results["test_r2"]:.4f}', fontweight='bold')
        axes[2,1].set_xlabel('Actual')
        axes[2,1].set_ylabel('Predicted')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.suptitle('XGBoost Split Files - All Sets Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('xgboost_split_files_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Comparison plot saved as 'xgboost_split_files_comparison.png'")
    
    def plot_performance_summary(self):
        """Plot tổng kết hiệu suất"""
        if self.results is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # R² comparison
        sets = ['Train', 'Validation', 'Test']
        r2_scores = [self.results['train_r2'], self.results['val_r2'], self.results['test_r2']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars1 = ax1.bar(sets, r2_scores, color=colors, alpha=0.8)
        ax1.set_title('R² Score Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('R² Score')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Error metrics
        error_metrics = ['MSE (×1000)', 'MAE (×1000)']
        test_errors = [self.results['test_mse']*1000, self.results['test_mae']*1000]
        
        bars2 = ax2.bar(error_metrics, test_errors, color='red', alpha=0.7)
        ax2.set_title('Test Set Error Metrics', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Error Value')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, error in zip(bars2, test_errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('XGBoost Split Files - Performance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('xgboost_split_files_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Performance summary saved as 'xgboost_split_files_summary.png'")

def main():
    print("🚀 XGBoost Split Files Model")
    print("=" * 60)
    print("Configuration:")
    print("  • Đọc từ: train_data.csv, val_data.csv, test_data.csv")
    print("  • Features: 4 basic features only")
    print("  • Window Size: 16")
    print("  • Learning Rate: 0.1")
    print("  • Max Depth: 10")
    print("=" * 60)
    
    # Initialize predictor
    predictor = XGBoostSplitFilesPredictor(window_size=16)
    
    # Train model từ split files
    results = predictor.train_model_from_files()
    
    if results:
        print(f"\n✅ Model training từ split files hoàn thành!")
        
        # Create plots
        predictor.plot_all_comparison()
        predictor.plot_performance_summary()
        
        print(f"\n🎯 Final Performance:")
        print(f"   Test R² Score: {results['test_r2']:.4f}")
        print(f"   Test MSE: {results['test_mse']:.6f}")
        print(f"   Test MAE: {results['test_mae']:.6f}")
        
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    main() 