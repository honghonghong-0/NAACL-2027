#!/usr/bin/env python3
"""Train SASRec on ReDial-MS and export validation scores in Pool-A format."""

import argparse
import collections
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "scripts"))
from baselines_simple import B, IDX, N, P, movies, users  # noqa: E402

MAX_LEN = 50
DIM = 64


class SASRec(nn.Module):
    def __init__(self, n_items, dim=DIM, layers=2, heads=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, item_ids):
        hidden = self.dropout(
            self.embedding(item_ids) * math.sqrt(DIM)
            + self.position.weight[None, : item_ids.size(1)]
        )
        mask = torch.triu(
            torch.ones(item_ids.size(1), item_ids.size(1), dtype=torch.bool, device=item_ids.device),
            1,
        )
        hidden = self.norm(
            self.encoder(hidden, mask=mask, src_key_padding_mask=(item_ids == 0))
        )
        return hidden @ self.embedding.weight[1:].T


def pad(sequence):
    sequence = sequence[-MAX_LEN:]
    return [0] * (MAX_LEN - len(sequence)) + [IDX[item] + 1 for item in sequence]


def training_sequences(allowed_splits):
    output = []
    for sessions in users.values():
        sequence = [
            item
            for session in sessions
            if session["split"] in allowed_splits
            for item in movies(session["messages"])
        ]
        sequence = [item for index, item in enumerate(sequence) if index == 0 or item != sequence[index - 1]]
        for end in range(2, len(sequence) + 1):
            output.append(sequence[max(0, end - MAX_LEN - 1):end])
        if len(sequence) >= 2:
            output.append(sequence[-MAX_LEN - 1:])
    return output


def load_pool_mapping():
    mapping = json.load((P / "pool_a.json").open())
    pool = [int(item) for item in mapping["pool"]]
    pool_position = {item: index for index, item in enumerate(pool)}
    redial_indices = []
    pool_indices = []
    for redial_id, canonical_id in mapping["redial2canon"].items():
        canonical_id = int(canonical_id)
        if redial_id in IDX and canonical_id in pool_position:
            redial_indices.append(IDX[redial_id])
            pool_indices.append(pool_position[canonical_id])
    return mapping, np.asarray(pool, dtype=np.int64), np.asarray(redial_indices), np.asarray(pool_indices)


def score_validation(model, device, mapping, pool_items, redial_indices, pool_indices):
    sessions_by_key = {
        (int(user), session["conv_id"]): session
        for user, sessions in users.items()
        for session in sessions
    }
    turns = [json.loads(line) for line in (P / "eval_turns_valid.jsonl").open()]
    keys = []
    scores = []
    model.eval()
    with torch.no_grad():
        for turn in turns:
            session = sessions_by_key[(turn["user"], turn["conv_id"])]
            history = [
                item
                for past in users[str(turn["user"])]
                if past["split"] == "history" and past["conv_id"] != turn["conv_id"]
                for item in movies(past["messages"])
            ]
            prefix = movies(session["messages"][: turn["turn_index"]])
            sequence = history + prefix
            if sequence:
                redial_scores = (
                    model(torch.tensor([pad(sequence)], device=device))[0, -1]
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                redial_scores = np.zeros(N, dtype=np.float32)
            pool_scores = np.full(len(pool_items), -np.inf, dtype=np.float32)
            np.maximum.at(pool_scores, pool_indices, redial_scores[redial_indices])
            keys.append((turn["user"], turn["conv_id"], turn["turn_index"]))
            scores.append(pool_scores)
    return turns, np.asarray(keys, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def validation_ndcg10(turns, scores, mapping, pool_items):
    pool_position = {int(item): index for index, item in enumerate(pool_items.tolist())}
    redial_to_canonical = {str(key): int(value) for key, value in mapping["redial2canon"].items()}
    sessions_by_key = {
        (int(user), session["conv_id"]): session
        for user, sessions in users.items()
        for session in sessions
    }
    values = []
    tie_noise = np.random.default_rng(0).random(len(pool_items)) * 1e-7
    for turn, raw_scores in zip(turns, scores):
        session = sessions_by_key[(turn["user"], turn["conv_id"])]
        current = movies(session["messages"][: turn["turn_index"]])
        excluded = {
            pool_position[redial_to_canonical[item]]
            for item in current
            if item in redial_to_canonical and redial_to_canonical[item] in pool_position
        }
        targets = {
            pool_position[redial_to_canonical[item]]
            for item in turn["targets"]
            if item in redial_to_canonical and redial_to_canonical[item] in pool_position
        }
        if not targets:
            continue
        ranked_scores = raw_scores.astype(np.float64) + tie_noise
        if excluded:
            ranked_scores[list(excluded)] = -np.inf
        order = np.argsort(-ranked_scores)
        ranks = np.empty(len(pool_items), dtype=np.int64)
        ranks[order] = np.arange(1, len(pool_items) + 1)
        for target in targets:
            rank = int(ranks[target])
            values.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
    return float(np.mean(values)), len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sequences = training_sequences({"history"})
    inputs = torch.tensor([pad(sequence[:-1]) for sequence in sequences])
    labels = torch.tensor([pad(sequence[1:]) for sequence in sequences])
    model = SASRec(N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mapping, pool_items, redial_indices, pool_indices = load_pool_mapping()
    best_metric = -1.0
    best_epoch = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(inputs))
        epoch_loss = []
        for start in range(0, len(inputs), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            x = inputs[batch].to(device)
            y = labels[batch].to(device)
            logits = model(x).reshape(-1, N)
            targets = (y - 1).clamp(min=-100).masked_fill(y == 0, -100).reshape(-1)
            loss = nn.functional.cross_entropy(logits, targets, ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach()))

        if epoch % args.eval_every == 0:
            turns, _, pool_scores = score_validation(
                model, device, mapping, pool_items, redial_indices, pool_indices
            )
            ndcg10, sample_count = validation_ndcg10(turns, pool_scores, mapping, pool_items)
            print(
                f"seed={args.seed} epoch={epoch} loss={np.mean(epoch_loss):.6f} "
                f"poolA_NDCG@10={ndcg10:.6f} n={sample_count}",
                flush=True,
            )
            if ndcg10 > best_metric:
                best_metric = ndcg10
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    model.load_state_dict(best_state)
    turns, keys, pool_scores = score_validation(
        model, device, mapping, pool_items, redial_indices, pool_indices
    )
    selected_metric, sample_count = validation_ndcg10(turns, pool_scores, mapping, pool_items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keys=keys, scores=pool_scores, item_ids=pool_items)
    metadata = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_ndcg10": selected_metric,
        "n_targets": sample_count,
        "n_training_sequences": len(sequences),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
