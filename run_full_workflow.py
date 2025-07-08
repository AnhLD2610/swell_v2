#!/usr/bin/env python3
"""
Full Workflow: Split Data → Train Models → Compare Results
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)

def print_step(step_num, title):
    """Print formatted step"""
    print(f"\n📌 Bước {step_num}: {title}")
    print("-" * 60)

def check_prerequisites():
    """Check if final_data.csv exists"""
    if not Path('final_data.csv').exists():
        print("❌ Không tìm thấy 'final_data.csv'")
        print("   Vui lòng đảm bảo file này có trong thư mục hiện tại")
        return False
    print("✓ Tìm thấy 'final_data.csv'")
    return True

def run_split_data():
    """Run split_data.py"""
    print("🔄 Chia dữ liệu thành train/val/test sets...")
    
    try:
        from split_data import split_data_to_files
        result = split_data_to_files()
        print("✅ Chia data thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi chia data: {e}")
        return False

def run_original_training():
    """Run original XGBoost training"""
    print("🚀 Training Original XGBoost Model...")
    
    try:
        from xgboost_final_best import XGBoostFinalBestPredictor
        
        predictor = XGBoostFinalBestPredictor(window_size=16)
        df = predictor.load_data('final_data.csv')
        results = predictor.train_model(df)
        
        if results:
            print("✅ Original training hoàn thành!")
            return True
        else:
            print("❌ Original training failed")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi training original: {e}")
        return False

def run_split_files_training():
    """Run split files XGBoost training"""
    print("🚀 Training Split Files XGBoost Model...")
    
    try:
        from xgboost_split_files import XGBoostSplitFilesPredictor
        
        predictor = XGBoostSplitFilesPredictor(window_size=16)
        results = predictor.train_model_from_files()
        
        if results:
            print("✅ Split files training hoàn thành!")
            return True
        else:
            print("❌ Split files training failed")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi training split files: {e}")
        return False

def run_comparison():
    """Run comparison between two approaches"""
    print("📊 So sánh 2 approaches...")
    
    try:
        from compare_approaches import main as compare_main
        compare_main()
        print("✅ So sánh hoàn thành!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi so sánh: {e}")
        return False

def cleanup_plots():
    """Clean up old plot files"""
    plot_files = [
        'xgboost_train_timeseries.png',
        'xgboost_val_timeseries.png', 
        'xgboost_test_timeseries.png',
        'xgboost_scatter_comparison.png',
        'xgboost_residuals_analysis.png',
        'xgboost_feature_importance.png',
        'xgboost_performance_metrics.png',
        'xgboost_model_summary.png',
        'xgboost_split_files_comparison.png',
        'xgboost_split_files_summary.png',
        'approaches_comparison.png'
    ]
    
    removed = 0
    for file in plot_files:
        if Path(file).exists():
            os.remove(file)
            removed += 1
    
    if removed > 0:
        print(f"🧹 Cleaned up {removed} old plot files")

def show_generated_files():
    """Show all generated files"""
    print_header("FILES ĐƯỢC TẠO")
    
    data_files = ['train_data.csv', 'val_data.csv', 'test_data.csv']
    plot_files = [
        'xgboost_train_timeseries.png',
        'xgboost_val_timeseries.png', 
        'xgboost_test_timeseries.png',
        'xgboost_scatter_comparison.png',
        'xgboost_residuals_analysis.png',
        'xgboost_feature_importance.png',
        'xgboost_performance_metrics.png',
        'xgboost_model_summary.png',
        'xgboost_split_files_comparison.png',
        'xgboost_split_files_summary.png',
        'approaches_comparison.png'
    ]
    
    print("📁 Data Files:")
    for file in data_files:
        if Path(file).exists():
            size = Path(file).stat().st_size / 1024 / 1024  # MB
            print(f"   ✓ {file} ({size:.2f} MB)")
        else:
            print(f"   ❌ {file}")
    
    print("\n🎨 Plot Files:")
    for file in plot_files:
        if Path(file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ❌ {file}")

def main():
    """Run full workflow"""
    start_time = time.time()
    
    print_header("XGBOOST SPLIT FILES - FULL WORKFLOW")
    print("Workflow này sẽ:")
    print("1. Chia final_data.csv thành train/val/test")
    print("2. Training Original XGBoost")
    print("3. Training Split Files XGBoost") 
    print("4. So sánh kết quả 2 approaches")
    
    # Cleanup old files
    cleanup_plots()
    
    # Step 1: Check prerequisites
    print_step(1, "Kiểm tra điều kiện")
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 2: Split data
    print_step(2, "Chia dữ liệu")
    if not run_split_data():
        print("❌ Không thể tiếp tục do lỗi chia data")
        sys.exit(1)
    
    # Step 3: Original training
    print_step(3, "Training Original Model")
    original_success = run_original_training()
    
    # Step 4: Split files training
    print_step(4, "Training Split Files Model")
    split_success = run_split_files_training()
    
    # Step 5: Compare if both succeeded
    if original_success and split_success:
        print_step(5, "So sánh kết quả")
        run_comparison()
    else:
        print("⚠️ Không thể so sánh do một trong hai training failed")
    
    # Show generated files
    show_generated_files()
    
    # Summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print_header("WORKFLOW HOÀN THÀNH")
    print(f"⏱️  Tổng thời gian: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    
    if original_success and split_success:
        print("✅ Cả 2 approaches đã được training và so sánh thành công!")
        print("📊 Kiểm tra file 'approaches_comparison.png' để xem kết quả so sánh")
    elif original_success:
        print("✅ Original approach training thành công")
        print("❌ Split files approach failed")
    elif split_success:
        print("❌ Original approach failed")
        print("✅ Split files approach training thành công")
    else:
        print("❌ Cả 2 approaches đều failed")
    
    print("\n🎯 Các files quan trọng:")
    print("   📁 train_data.csv, val_data.csv, test_data.csv")
    print("   📊 approaches_comparison.png")
    print("   🎨 Các plot files khác")

if __name__ == "__main__":
    main() 