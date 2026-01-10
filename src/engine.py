import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast

def train_one_epoch(model, loader, optimizer, scaler, device, grad_accum_steps=1, mixup_fn=None, criterion=None):
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
        
        # APPLY MIXUP ===
        if mixup_fn is not None:
            # Mixup input và transform label thành soft target
            videos, labels = mixup_fn(videos, labels)
        
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(videos)
            # Dùng criterion được truyền vào (SoftTargetCrossEntropy nếu có mixup)
            if criterion is not None:
                loss = criterion(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

        # Tính accuracy (chỉ mang tính tham khảo khi dùng Mixup vì label đã bị trộn)
        # Nếu dùng mixup, ta lấy argmax của prediction so với argmax của soft label
        if mixup_fn is not None:
            preds = logits.argmax(dim=1)
            gt_labels = labels.argmax(dim=1) # Lấy lại hard label từ soft label
            correct += (preds == gt_labels).sum().item()
        else:
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
        total += videos.size(0)

        loss = loss / grad_accum_steps
        # Xử lý Scaler
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches):
                optimizer.step()
                optimizer.zero_grad()

        loss_val = loss.item() * grad_accum_steps
        total_loss += loss_val * videos.size(0)
        
        progress.set_postfix(loss=f"{loss_val:.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total

def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Val", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            logits = model(videos)
            
            # Criterion cho val thường là CrossEntropy chuẩn
            if criterion is not None:
                loss = criterion(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            
    return correct / total, total_loss / total