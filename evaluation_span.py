#!/usr/bin/env python3
"""
Span-level evaluation script for the SRL model.

Usage:
  python evaluation_span.py \
    --model_path path/to/model.pt \
    --config_path path/to/model.json \
    --dataset_root datasets/preprocessed/ \
    --dataset UP \
    [--batch_size 32] [--threshold 0.5] [--device cpu]
"""
import argparse
import json
import torch
from tqdm import tqdm

from model import SRL_MODEL
from train.utils import get_dataloaders
from train.functions import get_spans


def extract_gold_spans(role_label_tensor: torch.Tensor) -> list[tuple]:
    """
    Extract gold argument spans from a role label tensor.

    Each span is represented as a tuple:
        (relation_index, role_index, start_token, end_token)

    Args:
        role_label_tensor: Tensor of shape [n_relations, n_tokens, n_roles], binary labels.
    Returns:
        List of span tuples.
    """
    spans = []
    # Convert to numpy for easy iteration
    labels = role_label_tensor.detach().cpu().numpy()
    n_rel, n_tokens, n_roles = labels.shape
    for rel_idx in range(n_rel):
        for role_idx in range(n_roles):
            in_span = False
            start = 0
            for tok_idx in range(n_tokens):
                if labels[rel_idx, tok_idx, role_idx] == 1:
                    if not in_span:
                        in_span = True
                        start = tok_idx
                else:
                    if in_span:
                        # close span
                        end = tok_idx - 1
                        spans.append((rel_idx, role_idx, start, end))
                        in_span = False
            # handle span reaching sequence end
            if in_span:
                end = n_tokens - 1
                spans.append((rel_idx, role_idx, start, end))
    return spans


def main():
    parser = argparse.ArgumentParser(description="Span-level evaluation of SRL model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to the model config (.json). Defaults to replacing .pt with .json in model_path.")
    parser.add_argument("--dataset_root", type=str, default="datasets/preprocessed/",
                        help="Root directory for preprocessed datasets")
    parser.add_argument("--dataset", choices=["UP", "NOM"], required=True,
                        help="Dataset name: UP or NOM")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for span prediction")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cpu or cuda). Defaults to cuda if available.")
    args = parser.parse_args()

    # Resolve config path if not provided
    config_path = args.config_path or args.model_path.rsplit('.', 1)[0] + '.json'
    # Load model configuration
    with open(config_path, 'r') as cfg_file:
        config = json.load(cfg_file)
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    # Initialize and load model
    model = SRL_MODEL(**config)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Prepare data loader
    _, _, test_loader, _, _ = get_dataloaders(
        args.dataset_root,
        batch_size=args.batch_size,
        shuffle=False,
        model_name=config.get('model_name', None),
        dataset=args.dataset,
    )

    total_predicted = 0
    total_gold = 0
    total_correct = 0

    # Iterate over test set
    for batch in tqdm(test_loader, desc="Evaluating spans"):
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        relations = batch['relation_position']  # list of lists
        word_ids = batch['word_ids']             # list of lists
        gold_labels = batch['role_labels']       # list of tensors

        # Model forward
        with torch.no_grad():
            _, _, role_results = model(
                input_ids,
                attention_masks,
                relations,
                word_ids,
            )

        # Per-sample span extraction and counting
        for idx in range(len(role_results)):
            logits = role_results[idx].detach().cpu()     # [n_rel, n_tokens, n_roles]
            labels = gold_labels[idx].detach().cpu()      # [n_rel, n_tokens, n_roles]

            # Predicted spans
            pred_spans = []
            n_rel, _, _ = logits.shape
            for rel_i in range(n_rel):
                spans_dict = get_spans(logits[rel_i], threshold=args.threshold)
                for role_i, token_idxs in spans_dict.items():
                    if not token_idxs:
                        continue
                    start = token_idxs[0]
                    end = token_idxs[-1]
                    pred_spans.append((rel_i, role_i, start, end))

            # Gold spans
            gold_spans = extract_gold_spans(labels)

            # Update counts
            total_predicted += len(pred_spans)
            total_gold += len(gold_spans)
            for span in pred_spans:
                if span in gold_spans:
                    total_correct += 1

    # Compute metrics
    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Report
    print(f"\nSpan-level Evaluation Results:")
    print(f"  Total gold spans     : {total_gold}")
    print(f"  Total predicted spans: {total_predicted}")
    print(f"  Correct spans        : {total_correct}")
    print(f"  Precision            : {precision:.4f}")
    print(f"  Recall               : {recall:.4f}")
    print(f"  F1-score             : {f1:.4f}")


if __name__ == '__main__':
    main()