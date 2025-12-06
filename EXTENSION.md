# Proposed Extension: Multi-View Temporal Ensembling for Video-LLMs

## 1. Problem Statement
The current TOPA architecture employs a **fixed uniform sampling strategy** during inference, where a video of any length is compressed into a fixed number of frames (default: `max_feats=10`). This creates a significant information bottleneck, particularly for long videos (e.g., EgoSchema clips ~3 minutes).

- **Data Loss:** For a 5400-frame video, sampling only 10 frames results in a **99.8% loss of visual information**.
- **Temporal Blind Spots:** The sampling stride (gap between frames) becomes very large (~18 seconds for a 3-minute video). Critical actions or transitions occurring between these sampled frames are completely invisible to the model.
- **Rigidity:** The model's `temporal_emb` layer is a fixed-size learned matrix (10x4096), preventing us from simply increasing the number of input frames at inference time without retraining or complex interpolation.

## 2. Proposed Solution: Multi-View Temporal Ensembling
We propose a **training-free inference strategy** that aggregates information from multiple temporal "views" of the video. Instead of a single pass with one set of sampled frames, we perform multiple forward passes with different sampling offsets and ensemble the results.

### Methodology
1.  **View Generation:** For a video $V$ with length $L$ and target frame count $M$ (`max_feats`), the user specifies $K$ distinct views.
    - **Stride Calculation:** We define the sampling stride (or chunk length) as $S = L / M$.
    - **Dynamic View Adjustment:** $K$ is pre-determined by the user. We only reduce it if the stride is too small:
        - If $S < K$, we set $K' = \lfloor S \rfloor$ (or 1 if $S < 1$). Otherwise, $K' = K$.
        - This ensures we don't create redundant identical views for short videos.
    - **Offset Calculation:** For each view $k \in \{0, ..., K'-1\}$, we calculate a sampling offset $\delta_k$:
        $$ \delta_k = \lfloor k \times \frac{S}{K'} \rfloor $$
    - **Sampling:** The frame index for the $j$-th feature in view $k$ is:
        $$ \text{Index}_{j,k} = \min(\lfloor j \times S + \delta_k \rfloor, L-1) $$

2.  **Parallel Inference:** We feed each view $v_k$ into the frozen TOPA model independently.
    - The model produces a set of logits (prediction scores) $L_k$ for the answer options.

3.  **Logit Averaging:** The final prediction is derived from the average of logits across all valid views:
    $$ L_{final} = \frac{1}{K'} \sum_{k=1}^{K'} L_k $$

### Justification & Impact
- **Improved Temporal Coverage:** By using 5 views of 10 frames each, the model effectively "sees" 50 unique frames, reducing the temporal gap between observations from ~18s to ~3.6s.
- **Robustness:** Reduces the risk of "unlucky" sampling where key visual evidence falls between frames.
- **Zero Training Cost:** This method requires **no changes to the model weights** and **no re-training**. It leverages the existing pre-trained capabilities.
- **Architectural Compatibility:** It respects the fixed `temporal_emb` size (10 frames) by processing standard-sized inputs in multiple passes, avoiding the "matrix shape mismatch" problem.

## 3. Implementation Plan

### 3.1 Modify Dataloaders
Add `view_index` and `num_views` parameters to dataset classes for deterministic shifted sampling.

**File: `dataloader/nextqa.py`**

```python
# --- MODIFICATION 1: Add view_index and num_views parameters to __init__ ---
def __init__(self, args=None, tokenizer=None, split='train', view_index=0, num_views=1):
    super().__init__(args, tokenizer, split)
    self.split = split
    self.view_index = view_index    # Current view k (0 to K-1)
    self.num_views = num_views      # User's requested K
    self.data = pd.read_csv(f'./data/nextqa/{split}.csv')
    # ... rest unchanged ...

# --- MODIFICATION 2: Update _get_video with Multi-View logic ---
def _get_video(self, video):
    video = video / video.norm(dim=-1, keepdim=True)
    
    if len(video) > self.max_feats:
        # ========== NEW: Multi-View Temporal Ensembling Logic ==========
        L = len(video)
        M = self.max_feats
        S = L / M  # Float stride
        
        # Dynamic View Adjustment: Only reduce K if S < requested K
        K = self.num_views
        if S < K:
            K_prime = max(1, int(S))  # Reduce to floor(S)
        else:
            K_prime = K  # Use user's requested K
        
        # Clamp view_index to valid range
        k = self.view_index % K_prime
        
        # Calculate frame offset (delta_k) for this specific view
        delta_k = int(k * (S / K_prime))
        
        sampled = []
        for j in range(self.max_feats):
            # Calculate index: floor(j * S + delta_k)
            idx = min(int(j * S + delta_k), L - 1)
            sampled.append(video[idx])
        # ========== END NEW ==========
        video = torch.stack(sampled)
        video_len = self.max_feats
    elif len(video) < self.max_feats:
        # ... rest unchanged ...
```

### 3.2 Create Inference Script
**File: `multiview_inference.py`** (NEW FILE)

```python
"""
Multi-View Temporal Ensembling Inference Script

This script performs inference with multiple temporal views and aggregates
the results via logit averaging.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from dataloader.nextqa import NextQA
from llama import load_model  # Adjust based on actual model loading

def run_multiview_inference(args):
    # Load model
    model = load_model(args.checkpoint)
    model.eval()
    
    num_views = args.num_views
    all_logits = []
    
    # Run inference for each view
    for view_idx in range(num_views):
        print(f"Processing view {view_idx + 1}/{num_views}")
        
        # Create dataset with specific view_index and num_views
        dataset = NextQA(
            args=args,
            tokenizer=model.tokenizer,
            split='val',
            view_index=view_idx,   # <-- Current view k
            num_views=num_views    # <-- User's requested K
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        view_logits = []
        with torch.no_grad():
            for batch in dataloader:
                logits = model(batch)  # Get prediction logits
                view_logits.append(logits)
        
        all_logits.append(torch.cat(view_logits, dim=0))
    
    # ========== Logit Averaging ==========
    stacked_logits = torch.stack(all_logits, dim=0)  # [K, N, num_options]
    final_logits = stacked_logits.mean(dim=0)        # [N, num_options]
    predictions = final_logits.argmax(dim=-1)
    
    # Calculate accuracy
    # ... evaluation logic ...
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, default=5, help="Number of temporal views (K)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    # ... other args ...
    args = parser.parse_args()
    
    run_multiview_inference(args)
```

### 3.3 Summary of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `dataloader/nextqa.py` | Modified | Added `view_index`, `num_views` params and Multi-View sampling logic |
| `dataloader/textvid.py` | Modified | Same changes as nextqa.py |
| `dataloader/egoschema.py` | Modified | Same changes as nextqa.py |
| `multiview_inference.py` | **NEW** | Orchestrates multi-pass evaluation and logit aggregation |

## 4. Evaluation Plan
- **Datasets:** NextQA (validation), EgoSchema (test)
- **Metrics:** Top-1 Accuracy
- **Comparison:** Single-view baseline (view_offset=0) vs Multi-View (K=5)
- **Ablation:** Vary K ∈ {1, 3, 5, 7, 10} to study the effect of view count
