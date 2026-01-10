import argparse
import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import TestDataset, test_collate_fn
from src.model import LSViTForAction
from src.config import ModelConfig

# === DANH SÁCH CHUẨN 51 CLASS HMDB51 (A-Z) ===
# Bắt buộc phải khớp với thứ tự lúc train (ImageFolder tự sort theo tên)
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

def save_submission(ids: list, predictions: list, submission_file: Path) -> Path:
    """Save predictions to CSV file correctly."""
    
    # Tạo DataFrame từ ID thực tế và Tên Class dự đoán
    df = pd.DataFrame({'id': ids, 'class': predictions})
    
    # Sắp xếp theo ID (để đảm bảo thứ tự 0, 1, 2...)
    # Chuyển id sang int để sort đúng (tránh 1, 10, 2...), sau đó sort
    try:
        df['id_num'] = df['id'].astype(int)
        df = df.sort_values(by='id_num').drop(columns=['id_num'])
    except:
        # Nếu ID không phải số thì sort string bình thường
        df = df.sort_values(by='id')

    # Check môi trường Kaggle
    is_kaggle_env = os.path.exists('/kaggle/working')
    
    if is_kaggle_env:
        output_path = Path('/kaggle/working/submission.csv')
    else:
        output_path = submission_file

    df.to_csv(output_path, index=False)
    print(f"\n✓ Submission file created at: {output_path}")
    print("Preview:")
    print(df.head())
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on test set')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint (e.g., best_model.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to test data directory containing folders 0, 1, 2...')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to sample')
    parser.add_argument('--frame_stride', type=int, default=2,
                        help='Stride between frames')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for model input')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--submission_file', type=str, default='submission.csv',
                        help='Path to save submission file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print("INFERENCE ON TEST SET")

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Config Model
    model_config = ModelConfig()
    model_config.num_classes = 51  # HMDB51 luôn là 51
    model_config.image_size = args.image_size

    # Initialize model
    model = LSViTForAction(config=model_config).to(device)

    # Load weights (Xử lý thông minh các trường hợp key khác nhau)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Clean keys: Xóa prefix 'module.' hoặc '_orig_mod.'
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith('module.'): clean_k = clean_k[7:]
        if clean_k.startswith('_orig_mod.'): clean_k = clean_k[10:]
        new_state_dict[clean_k] = v

    # Load vào model
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed ({e}). Retrying with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    print("Model loaded successfully!")

    print("\nLoading test dataset...")
    # TestDataset trả về (video_tensor, video_id)
    test_dataset = TestDataset(
        root=args.data_root,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        image_size=args.image_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn # Quan trọng: collate trả về IDs
    )
    print(f"Test samples: {len(test_dataset)}")

    # Run inference
    print("\nRunning inference...")
    predictions = []
    video_ids = []

    with torch.no_grad():
        for batch_idx, (videos, ids) in enumerate(test_loader):
            videos = videos.to(device)
            
            # Forward pass
            logits = model(videos)
            preds = logits.argmax(dim=1)

            # Lưu kết quả
            predictions.extend(preds.cpu().numpy())
            video_ids.extend(ids) # ids thường là tuple hoặc list string

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples")

    print(f"\nInference complete! Processed {len(predictions)} videos")

    # === MAPPING: SỐ -> TÊN CLASS (Quan trọng cho Submission) ===
    predicted_class_names = []
    for pred_idx in predictions:
        if 0 <= pred_idx < len(HMDB51_CLASSES):
            predicted_class_names.append(HMDB51_CLASSES[pred_idx])
        else:
            # Fallback nếu model dự đoán ra ngoài range (hiếm gặp)
            predicted_class_names.append("unknown")

    # Create submission
    submission_path = Path(args.submission_file)
    saved_path = save_submission(video_ids, predicted_class_names, submission_path)

    return saved_path

if __name__ == "__main__":
    main()