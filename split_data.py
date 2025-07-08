import pandas as pd
import numpy as np
import os

def split_data_to_files(input_file='final_data.csv', test_split=0.05, val_split=0.05):
    """
    Chia final_data.csv thành train.csv, val.csv, test.csv theo thứ tự thời gian
    """
    print(f"🔄 Chia file {input_file} thành train/val/test sets...")
    
    # Đọc data
    print(f"📖 Đọc dữ liệu từ {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Kích thước gốc: {len(df)} rows")
    
    # Sắp xếp theo thời gian
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"✓ Dữ liệu đã sắp xếp theo thời gian")
        print(f"   Từ: {df['datetime'].iloc[0]}")
        print(f"   Đến: {df['datetime'].iloc[-1]}")
    else:
        print("⚠️ Không tìm thấy cột 'datetime', sử dụng thứ tự hiện tại")
    
    # Tính toán index chia
    train_split = 1 - test_split - val_split
    
    train_idx = int(len(df) * train_split)
    val_idx = int(len(df) * (train_split + val_split))
    
    # Chia data
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    # Thêm slice cho test set như trong code gốc
    if len(test_df) > 9:  # Đảm bảo có đủ data để slice
        test_df = test_df.iloc[3:-6].copy()
    
    print(f"\n📊 Kết quả chia:")
    print(f"   Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    if 'datetime' in df.columns:
        print(f"\n📅 Phạm vi thời gian:")
        print(f"   Train: {train_df['datetime'].iloc[0]} → {train_df['datetime'].iloc[-1]}")
        print(f"   Val:   {val_df['datetime'].iloc[0]} → {val_df['datetime'].iloc[-1]}")
        print(f"   Test:  {test_df['datetime'].iloc[0]} → {test_df['datetime'].iloc[-1]}")
    
    # Lưu files
    train_file = 'train_data.csv'
    val_file = 'val_data.csv'
    test_file = 'test_data.csv'
    
    print(f"\n💾 Lưu files:")
    
    train_df.to_csv(train_file, index=False)
    print(f"   ✓ {train_file}")
    
    val_df.to_csv(val_file, index=False)
    print(f"   ✓ {val_file}")
    
    test_df.to_csv(test_file, index=False)
    print(f"   ✓ {test_file}")
    
    # Thống kê target
    target_col = 'arb_profit'
    if target_col in df.columns:
        print(f"\n📈 Thống kê target '{target_col}':")
        print(f"   Train: mean={train_df[target_col].mean():.6f}, std={train_df[target_col].std():.6f}")
        print(f"   Val:   mean={val_df[target_col].mean():.6f}, std={val_df[target_col].std():.6f}")
        print(f"   Test:  mean={test_df[target_col].mean():.6f}, std={test_df[target_col].std():.6f}")
    
    return {
        'train_file': train_file,
        'val_file': val_file, 
        'test_file': test_file,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }

if __name__ == "__main__":
    # Chia data
    result = split_data_to_files()
    
    print(f"\n✅ Hoàn thành! Đã tạo 3 files:")
    print(f"   📁 {result['train_file']}")
    print(f"   📁 {result['val_file']}")
    print(f"   📁 {result['test_file']}")
    print(f"\nBây giờ có thể sử dụng các file này để training!") 