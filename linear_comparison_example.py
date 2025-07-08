from linear_final_best import LinearRegressionFinalBestPredictor
import pandas as pd

def compare_scaling_options():
    """Compare models with and without feature scaling"""
    
    print("🔬 Comparing Linear Regression Models: With vs Without MinMax Scaling")
    print("=" * 80)
    
    # Load data once
    df = pd.read_csv('final_data.csv')
    
    configs = [
        {
            'name': 'Linear Regression with MinMax Scaling',
            'window_size': 16,
            'model_type': 'linear',
            'use_scaling': True
        },
        {
            'name': 'Linear Regression without Scaling',
            'window_size': 16,
            'model_type': 'linear',
            'use_scaling': False
        },
        {
            'name': 'Ridge Regression with MinMax Scaling',
            'window_size': 16,
            'model_type': 'ridge',
            'use_scaling': True
        },
        {
            'name': 'Ridge Regression without Scaling',
            'window_size': 16,
            'model_type': 'ridge',
            'use_scaling': False
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Create predictor with current config
        predictor = LinearRegressionFinalBestPredictor(
            window_size=config['window_size'],
            model_type=config['model_type'],
            use_scaling=config['use_scaling'],
            filter_outliers=True  # Use outlier filtering for comparison
        )
        
        # Train model
        result = predictor.train_model(df)
        
        # Store results
        results[config['name']] = {
            'train_r2': result['train_r2'],
            'val_r2': result['val_r2'],
            'test_r2': result['test_r2'],
            'test_mse': result['test_mse'],
            'test_mae': result['test_mae'],
            'config': config
        }
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'Train R²':<10} {'Val R²':<10} {'Test R²':<10} {'Test MSE':<12} {'Test MAE':<10}")
    print("-" * 85)
    
    for name, result in results.items():
        print(f"{name:<35} {result['train_r2']:<10.4f} {result['val_r2']:<10.4f} "
              f"{result['test_r2']:<10.4f} {result['test_mse']:<12.6f} {result['test_mae']:<10.6f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n🏆 Best Model (by Test R²): {best_model[0]}")
    print(f"   Test R² = {best_model[1]['test_r2']:.4f}")
    
    return results

def quick_test_single_config():
    """Quick test with a single configuration"""
    
    print("⚡ Quick Test - Single Configuration")
    print("=" * 50)
    
    # You can easily change these parameters:
    WINDOW_SIZE = 20
    MODEL_TYPE = 'ridge'  # 'linear', 'ridge', or 'lasso'
    USE_SCALING = True    # True or False
    
    predictor = LinearRegressionFinalBestPredictor(
        window_size=WINDOW_SIZE,
        model_type=MODEL_TYPE,
        use_scaling=USE_SCALING,
        filter_outliers=False  # No filtering to see real performance
    )
    
    # Load data and train
    df = predictor.load_data('final_data.csv')
    results = predictor.train_model(df)
    
    if results:
        print(f"\n✅ Training completed!")
        print(f"   Configuration: {MODEL_TYPE.title()} Regression, Window={WINDOW_SIZE}, MinMax Scaling={'On' if USE_SCALING else 'Off'}")
        print(f"   Test R² = {results['test_r2']:.4f}")
        
        # Optionally create plots
        create_plots = input("\n📈 Create analysis plots? (y/n): ").lower().strip()
        if create_plots == 'y':
            predictor.create_all_plots()

if __name__ == "__main__":
    choice = input("Choose option:\n1. Compare scaling options\n2. Quick single test\nEnter (1 or 2): ").strip()
    
    if choice == "1":
        compare_scaling_options()
    elif choice == "2":
        quick_test_single_config()
    else:
        print("Invalid choice. Running comparison by default...")
        compare_scaling_options() 