import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast

def train_one_epoch(model, loader, optimizer, scaler, device, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    optimizer.zero_grad()
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == 'mps':
       
        use_amp = True 
    elif device_type == 'cuda':
        use_amp = True
    else:
        use_amp = False
    
    progress = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (videos, labels) in enumerate(progress):
        videos, labels = videos.to(device), labels.to(device)
        
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(videos)
            loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss = loss / grad_accum_steps
        
        # Xử lý Scaler
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Nếu không dùng AMP (ví dụ CPU hoặc MPS tắt AMP), chạy backward thường
            loss.backward()
            if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches):
                optimizer.step()
                optimizer.zero_grad()

        loss_val = loss.item() * grad_accum_steps
        total_loss += loss_val * videos.size(0)
        
        progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Val", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            logits = model(videos)
            loss = F.cross_entropy(logits, labels)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
    return correct / total, total_loss / total