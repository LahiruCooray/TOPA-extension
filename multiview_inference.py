"""
Multi-View Temporal Ensembling Inference Script

This script performs inference with multiple temporal views and aggregates
the results via logit averaging. This is a training-free extension that
improves temporal coverage for long videos.

Usage:
    python multiview_inference.py --model 7B --dataset nextqa --num_views 5 \
        --resume vqa_checkpoint/checkpoint_pretrain/llama2_7b.../checkpoint_19.pth \
        --llama2 --llama_model_path ./pretrained/llama2/
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import util.misc as misc
from llama import Tokenizer, Tokenizer_llama3
from llama_vqa import LLaMA_VQA
from dataloader import num_options_mapping
from dataloader.nextqa import NextQA
from dataloader.egoschema import EgoSchema


def get_args_parser():
    parser = argparse.ArgumentParser('Multi-View Temporal Ensembling Inference', add_help=False)
    
    # ========== NEW: Multi-View Parameters ==========
    parser.add_argument('--num_views', default=5, type=int, 
                        help='Number of temporal views for ensembling (default: 5)')
    
    # Model parameters (same as train.py)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--llama_model_path', default='./pretrained/llama/', type=str)
    parser.add_argument('--model', default='7B', type=str)
    parser.add_argument('--adapter_layer', type=int, default=32)
    parser.add_argument('--adapter_len', type=int, default=50)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--max_feats', type=int, default=10)
    
    # Dataset parameters
    parser.add_argument('--dataset', default='nextqa', type=str)
    parser.add_argument('--output_dir', default='./results/multiview', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    
    # Distributed parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)
    
    # Model variants
    parser.add_argument('--llama2', action='store_true')
    parser.add_argument('--llama3', action='store_true')
    
    # Evaluation options
    parser.add_argument('--bias', type=float, default=3.)
    parser.add_argument('--tau', type=float, default=100.)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--sub', action='store_true')
    parser.add_argument('--openvqa_eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single_frame', action='store_true')
    parser.add_argument('--split', default='val', type=str, choices=['val', 'test'])
    
    # Unused but required by model
    parser.add_argument('--vaq', action='store_true')
    parser.add_argument('--qav', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--data_ratio', type=float, default=1.)
    parser.add_argument('--textvid', action='store_true')
    parser.add_argument('--variance', type=float, default=0.)
    parser.add_argument('--video_caption', action='store_true')
    parser.add_argument('--instruct', action='store_true')
    parser.add_argument('--openvqa', action='store_true')
    parser.add_argument('--weight_captioning', type=float, default=1.0)
    parser.add_argument('--webvid', action='store_true')
    parser.add_argument('--answer_balance', action='store_true')
    parser.add_argument('--accum_iter', default=1, type=int)
    
    return parser


def load_dataset_with_view(args, tokenizer, split, view_index, num_views):
    """
    Load dataset with a specific view index for Multi-View ensembling.
    
    Args:
        view_index: The current view index k (0 to num_views-1)
        num_views: User's requested K (total number of views)
    """
    dataset_mapping = {
        'nextqa': NextQA,
        'egos': EgoSchema,
    }
    
    if args.dataset not in dataset_mapping:
        raise ValueError(f"Dataset {args.dataset} not supported for Multi-View inference. "
                         f"Supported: {list(dataset_mapping.keys())}")
    
    dataset_cls = dataset_mapping[args.dataset]
    dataset = dataset_cls(args=args, tokenizer=tokenizer, split=split, 
                          view_index=view_index, num_views=num_views)
    
    return dataset


def run_single_view_inference(model, dataloader, args):
    """
    Run inference for a single view and return the raw logits.
    """
    model.eval()
    all_logits = []
    all_answers = []
    all_qids = []
    
    with torch.no_grad():
        for data in dataloader:
            answer = data['answer'].cuda()
            qid = data['qid']
            
            mode = 'caption' if args.openvqa_eval else 'vqa'
            logits = model(data, inference=True, mode=mode)
            
            all_logits.append(logits.cpu())
            all_answers.append(answer.cpu())
            all_qids.extend(qid if isinstance(qid, list) else qid.tolist())
    
    return torch.cat(all_logits, dim=0), torch.cat(all_answers, dim=0), all_qids


def multiview_ensemble(args):
    """
    Main function for Multi-View Temporal Ensembling inference.
    """
    misc.init_distributed_mode(args)
    
    print('=' * 60)
    print('Multi-View Temporal Ensembling Inference')
    print(f'Number of views: {args.num_views}')
    print(f'Dataset: {args.dataset}')
    print(f'Split: {args.split}')
    print('=' * 60)
    
    device = torch.device(args.device)
    
    # Fix seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # Load tokenizer
    if args.llama3:
        tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}./tokenizer.model')
    else:
        tokenizer = Tokenizer(model_path=f'{args.llama_model_path}./tokenizer.model')
    
    # Set num_options for the dataset
    args.num_options = num_options_mapping[args.dataset]
    
    # Load model
    model = LLaMA_VQA(args)
    model.to(device)
    model.re_init_freqs(600)
    
    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded checkpoint from: {args.resume}")
    
    # ========== Multi-View Ensembling Loop ==========
    all_view_logits = []
    
    for view_idx in range(args.num_views):
        print(f"\n>>> Processing View {view_idx + 1}/{args.num_views} (view_index={view_idx})")
        
        # Load dataset with specific view_index and total num_views (user's K)
        dataset = load_dataset_with_view(args, tokenizer, args.split, 
                                         view_index=view_idx, num_views=args.num_views)
        
        # Create dataloader (no distributed sampler for deterministic ordering)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=batch_collate
        )
        
        # Run inference for this view
        logits, answers, qids = run_single_view_inference(model, dataloader, args)
        all_view_logits.append(logits)
        
        # Calculate accuracy for this single view
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)
        view_acc = (prediction == answers).float().mean().item() * 100
        print(f"    View {view_idx + 1} Accuracy: {view_acc:.2f}%")
    
    # ========== Logit Averaging ==========
    print("\n>>> Ensembling logits across all views...")
    stacked_logits = torch.stack(all_view_logits, dim=0)  # [K, N, num_options, seq_len]
    ensemble_logits = stacked_logits.mean(dim=0)          # [N, num_options, seq_len]
    
    # Final prediction
    count = (ensemble_logits != 0).sum(-1)
    final_prediction = (ensemble_logits.sum(-1) / count).argmin(-1)
    
    ensemble_acc = (final_prediction == answers).float().mean().item() * 100
    
    print('=' * 60)
    print(f"FINAL ENSEMBLE ACCURACY ({args.num_views} views): {ensemble_acc:.2f}%")
    print('=' * 60)
    
    # Save results
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        results = {
            'dataset': args.dataset,
            'split': args.split,
            'num_views': args.num_views,
            'ensemble_accuracy': ensemble_acc,
            'checkpoint': args.resume,
        }
        with open(os.path.join(args.output_dir, 'multiview_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_dir}/multiview_results.json")
    
    return ensemble_acc


def batch_collate(batch):
    """
    Collate function for dataloader (copied from dataloader/__init__.py)
    """
    bs = len(batch)
    vid = [batch[i]["vid"] for i in range(bs)]
    video = torch.stack([batch[i]["video"] for i in range(bs)])
    video_len = torch.tensor([batch[i]["video_len"] for i in range(bs)], dtype=torch.long)
    text = [batch[i]["text"] for i in range(bs)]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer = torch.tensor([batch[i]["answer"] for i in range(bs)], dtype=torch.long)
    qtype = torch.tensor([batch[i]["qtype"] for i in range(bs)], dtype=torch.long) if "qtype" in batch[0] else None
    
    text_id = {}
    label = {}
    video_start = {}
    video_index = {}
    label_mask = {}
    
    for key in batch[0]["text_id"].keys():
        text_id[key] = torch.stack([batch[i]["text_id"][key] for i in range(bs)])
        label[key] = torch.stack([batch[i]["label"][key] for i in range(bs)])
        video_start[key] = torch.tensor([batch[i]["video_start"][key] for i in range(bs)], dtype=torch.long)
        video_index[key] = torch.stack([batch[i]["video_index"][key] for i in range(bs)])
        label_mask[key] = torch.stack([batch[i]["label_mask"][key] for i in range(bs)])
    
    return {
        "vid": vid, "video": video, "video_len": video_len, "text": text,
        "text_id": text_id, "label": label, "video_start": video_start,
        "video_index": video_index, "label_mask": label_mask, 
        "qid": qid, "answer": answer, "qtype": qtype
    }


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    multiview_ensemble(args)
