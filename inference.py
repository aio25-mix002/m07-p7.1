import argparse
import os
import torch
import torch.nn as nn  # <--- 1. CẦN THÊM IMPORT NÀY
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import các module từ source code của bạn
from src.dataset import TestDataset, test_collate_fn
from src.model import LSViTForAction
from src.config import ModelConfig

# === DANH SÁCH CHUẨN 51 CLASS HMDB51 (A-Z) ===
HMDB51_CLASSES = [
    'brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 
    'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble', 
    'drink', 'eat', 'fall_floor', 'fencer', 'flic_flac', 
    'golf', 'handstand', 'hit', 'hug', 'jump', 
    'kick', 'kick_ball', 'kiss', 'laugh', 'pick', 
    'pour', 'pullup', 'punch', 'push', 'pushup', 
    'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 
    'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 
    'smoke', 'somersault', 'stand', 'swing_baseball', 'sword', 
    'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave'
]

def save_submission(ids: list, predictions: list, submission_file: Path):
    """Lưu file submission đúng định dạng Kaggle yêu cầu."""
    df = pd.DataFrame({
        'id': ids,
        'class': predictions
    })
    
    try:
        df['id'] = df['id'].astype(int)
        df = df.sort_values(by='id').reset_index(drop=True)
    except ValueError:
        print("Cảnh báo: ID không phải dạng số, sẽ sort theo string.")
        df = df.sort_values(by='id').reset_index(drop=True)

    is_kaggle_env = os.path.exists('/kaggle/working')
    if is_kaggle_env:
        output_path = Path('/kaggle/working') / submission_file.name
    else:
        output_path = submission_file

    df.to_csv(output_path, index=False)
    print(f"\n✅ Đã tạo file submission tại: {output_path}")
    print("5 dòng đầu tiên của file:")
    print(df.head())
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn file best_model.pth')
    parser.add_argument('--data_root', type=str, required=True, help='Thư mục chứa folder test')
    parser.add_argument('--submission_file', type=str, default='submission.csv', help='Tên file kết quả đầu ra')
    
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Thiết lập thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load cấu hình và Model
    print("Khởi tạo model...")
    config = ModelConfig()
    config.num_classes = 51
    config.image_size = args.image_size
    
    model = LSViTForAction(config=config).to(device)

    # 3. Load Checkpoint
    # Lưu ý: Load weights TRƯỚC khi bật DataParallel để tránh lỗi key mismatch
    print(f"Loading weights từ: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[clean_k] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Lưu ý: Load strict thất bại, thử load lỏng (strict=False)...")
        model.load_state_dict(new_state_dict, strict=False)

    # === 2. THÊM ĐOẠN NÀY ĐỂ KÍCH HOẠT 2 GPU ===
    if torch.cuda.device_count() > 1:
        print(f"🚀 Kích hoạt chế độ Multi-GPU trên {torch.cuda.device_count()} cards!")
        model = nn.DataParallel(model)
    # ===========================================
    
    model.eval()

    # 4. Chuẩn bị dữ liệu Test
    print(f"Đọc dữ liệu từ: {args.data_root}")
    test_ds = TestDataset(
        root=args.data_root,
        num_frames=args.num_frames,
        frame_stride=2,
        image_size=args.image_size
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size, # Lưu ý: Khi dùng 2 GPU, nên tăng batch_size
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn
    )
    print(f"Tìm thấy {len(test_ds)} video clip.")

    # 5. Chạy Inference
    print("Bắt đầu dự đoán...")
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for videos, ids in tqdm(test_loader, desc="Processing"):
            videos = videos.to(device)
            
            # Forward pass (DataParallel tự chia batch sang các GPU)
            outputs = model(videos)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_ids.extend(ids)

    # 6. Map từ số sang tên Class
    print("Đang tạo file submission...")
    final_class_names = []
    for pred_idx in all_preds:
        if 0 <= pred_idx < len(HMDB51_CLASSES):
            final_class_names.append(HMDB51_CLASSES[pred_idx])
        else:
            final_class_names.append("unknown")

    # 7. Lưu file
    save_submission(all_ids, final_class_names, Path(args.submission_file))

if __name__ == '__main__':
    main()