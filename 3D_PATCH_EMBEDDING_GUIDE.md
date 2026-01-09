# 3D Patch Embedding Guide

This guide explains how to use 2D vs 3D patch embedding in the LSViT model.

## Overview

### 2D Patch Embedding (Default)
- **Input**: Video as sequence of frames
- **Processing**: Each frame is divided into 2D spatial patches
- **Patches**: `(H/P) × (W/P)` patches per frame
- **Total patches**: `T × (H/P) × (W/P)` for T frames
- **Use case**: Standard video classification, compatible with ImageNet pretrained weights

### 3D Patch Embedding (New)
- **Input**: Video as 3D volume
- **Processing**: Video divided into spatiotemporal cubes (tubelets)
- **Patches**: `(T/tubelet_size) × (H/P) × (W/P)` spatiotemporal patches
- **Total patches**: Reduced by factor of `tubelet_size`
- **Use case**: More efficient temporal modeling, captures motion in patches

---

## Configuration

### Using Environment Variables

```bash
# Enable 3D patch embedding
export APPCONFIG__USE_3D_PATCH_EMBED=true

# Set temporal patch size (tubelet size)
export APPCONFIG__TUBELET_SIZE=2  # Common values: 1, 2, 4

# Spatial patch size (same as before)
export APPCONFIG__PATCH_SIZE=16
```

### Using Code

```python
from src.config import ModelConfig

# 2D Patch Embedding (default)
config = ModelConfig()
config.use_3d_patch_embed = False
config.patch_size = 16

# 3D Patch Embedding
config = ModelConfig()
config.use_3d_patch_embed = True
config.patch_size = 16
config.tubelet_size = 2  # Temporal patch size
```

---

## Training

### With 2D Patches (Default)

```bash
python train.py --num_frames 16
```

### With 3D Patches

```bash
# Set environment variable
export APPCONFIG__USE_3D_PATCH_EMBED=true
export APPCONFIG__TUBELET_SIZE=2

# Run training
python train.py --num_frames 16
```

**Important**: `num_frames` must be divisible by `tubelet_size`!

---

## Inference

### With 2D Patches

```bash
python inference.py \
  --checkpoint ./model_2d.pth \
  --data_root ./data/test \
  --num_frames 16
```

### With 3D Patches

```bash
export APPCONFIG__USE_3D_PATCH_EMBED=true
export APPCONFIG__TUBELET_SIZE=2

python inference.py \
  --checkpoint ./model_3d.pth \
  --data_root ./data/test \
  --num_frames 16
```

**Note**: The checkpoint must match the patch embedding type used during training!

---

## Testing the Implementation

Run the test script to see the difference:

```bash
python test_3d_patch_embedding.py
```

This will:
1. Create sample video input
2. Process with 2D patch embedding
3. Process with 3D patch embedding
4. Compare the results

---

## Patch Count Comparison

### Example: 224×224 video with 16 frames

**2D Patch Embedding:**
- Patch size: 16×16
- Patches per frame: (224/16)² = 196
- Total patches: 16 × 196 = **3,136 patches**

**3D Patch Embedding (tubelet_size=2):**
- Spatial patches: (224/16)² = 196
- Temporal patches: 16/2 = 8
- Total patches: 8 × 196 = **1,568 patches**
- **Reduction: 50%** less patches!

**3D Patch Embedding (tubelet_size=4):**
- Spatial patches: (224/16)² = 196
- Temporal patches: 16/4 = 4
- Total patches: 4 × 196 = **784 patches**
- **Reduction: 75%** less patches!

---

## Technical Details

### 2D Conv vs 3D Conv

**2D Patch Embedding:**
```python
Conv2d(
    in_channels=3,
    out_channels=768,
    kernel_size=(16, 16),
    stride=(16, 16)
)
```

**3D Patch Embedding:**
```python
Conv3d(
    in_channels=3,
    out_channels=768,
    kernel_size=(2, 16, 16),  # (T, H, W)
    stride=(2, 16, 16)
)
```

### Input Shape Transformation

**2D:**
- Input: `(B, T, C, H, W)` → Reshape to `(B×T, C, H, W)`
- Patch embedding processes `B×T` frames
- Output: `(B×T, num_patches, embed_dim)`

**3D:**
- Input: `(B, T, C, H, W)` → Permute to `(B, C, T, H, W)`
- Patch embedding processes entire video
- Output: `(B, num_patches, embed_dim)`

---

## Advantages & Trade-offs

### 2D Patch Embedding

**Advantages:**
- ✅ Compatible with ImageNet pretrained weights
- ✅ Well-established approach
- ✅ Flexible frame count

**Disadvantages:**
- ❌ More patches to process
- ❌ Temporal modeling only in transformer blocks
- ❌ Slower inference

### 3D Patch Embedding

**Advantages:**
- ✅ Fewer patches (faster)
- ✅ Temporal information in patches
- ✅ More efficient for videos
- ✅ Better motion capture

**Disadvantages:**
- ❌ Cannot use ImageNet pretrained weights directly
- ❌ Requires training from scratch or video pretrained weights
- ❌ Frame count must be divisible by tubelet_size

---

## Best Practices

1. **Start with 2D** for baseline and transfer learning from ImageNet
2. **Use 3D** when:
   - Training from scratch on video data
   - Have sufficient video training data
   - Need faster inference
   - Motion is critical for the task

3. **Tubelet size selection:**
   - `tubelet_size=1`: Maximum temporal detail, same as 2D
   - `tubelet_size=2`: Good balance (recommended)
   - `tubelet_size=4`: Aggressive reduction, may lose temporal detail

4. **Frame count:**
   - Ensure `num_frames % tubelet_size == 0`
   - Common: `num_frames=16` with `tubelet_size=2` → 8 temporal patches

---

## References

- **ViViT**: Arnab et al., "ViViT: A Video Vision Transformer", ICCV 2021
- **TimeSformer**: Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?", ICML 2021
- **Video Swin Transformer**: Liu et al., "Video Swin Transformer", CVPR 2022

---

## Troubleshooting

### Error: "num_frames must be specified when using 3D patch embedding"

**Solution:** Pass `num_frames` when creating the model:
```python
model = LSViTForAction(config, num_frames=16)
```

### Error: Frame count not divisible by tubelet_size

**Solution:** Ensure `num_frames % tubelet_size == 0`:
- num_frames=16, tubelet_size=2 ✅
- num_frames=16, tubelet_size=4 ✅
- num_frames=16, tubelet_size=3 ❌

### Checkpoint incompatibility

2D and 3D models have different architectures and cannot share checkpoints directly.
Always train and infer with the same patch embedding type.

---

## Summary

3D patch embedding is a powerful alternative to 2D patches for video understanding:
- **Faster** due to fewer patches
- **Better** temporal modeling
- **Trade-off** with transfer learning capabilities

Choose based on your use case, data availability, and computational constraints.
