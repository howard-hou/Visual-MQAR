from typing import List

import numpy as np
import torch
from torch import Tensor


# Randomly generated dataset
def generate_dataset(
        vocab_size: int = 50,
        num_examples: int = 1000,
        seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    # Set the seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokens = torch.arange(vocab_size) + 1
    keys = tokens[: (vocab_size // 2) - 1]
    values = tokens[((vocab_size // 2) - 1):-1]
    examples = []
    repeated = []
    golds = []
    for _ in range(num_examples):
        # Random-shuffling
        shuffled_keys = keys[torch.randperm(len(keys))]
        shuffled_values = values[torch.randperm(len(values))]
        kv_pairs = list(zip(shuffled_keys, shuffled_values))
        example = kv_pairs[:4 * len(kv_pairs) // 8]
        example += kv_pairs[:4 * len(kv_pairs) // 8]
        example += kv_pairs[:4 * len(kv_pairs) // 8]
        # shuffle example and repeat_pos
        perm = torch.randperm(len(example))
        example = [example[i] for i in perm]

        # Collect the repeated tokens position.
        repeat_pos = []
        # Only used to store the repeated tokens
        repeats = {}
        for i, (k, v) in enumerate(example):
            if int(k) in repeats:
                repeat_pos.append(i)
            repeats[int(k)] = int(v)

        # example_tensor: shape [batch, seq_len]
        example_tensor = torch.tensor(example).flatten()
        repeat_tensor = [i * 2 for i in repeat_pos]

        # gold: extract the tokens from repeat tensor
        gold = []
        for i in repeat_tensor:
            gold.append(int(example_tensor[i + 1]))

        repeat_tensor = torch.tensor(repeat_tensor)
        gold_tensor = torch.tensor(gold)

        repeated.append(repeat_tensor)
        examples.append(example_tensor)
        golds.append(gold_tensor)
    examples = torch.stack(examples)
    repeated = torch.stack(repeated)
    golds = torch.stack(golds)
    return examples, repeated, golds


# Calculate the accuracy
def score_solution(
    repeated: torch.Tensor,
    predictions: torch.Tensor,
    golds: torch.Tensor,
) -> list[int]:
    correct = 0
    total = 0
    incorrect_indices = []
    for idx, (gold, repeat_positions, preds) in enumerate(zip(golds, repeated, predictions)):
        assert len(repeat_positions) == len(gold), \
            print(f"{idx}: {len(repeat_positions)} != {len(gold)} -- repeat: {repeat_positions}, gold: {gold}")
        num_correct = len([True for j, i in enumerate(repeat_positions) if gold[j] == preds[i]])
        correct += num_correct
        total += len(gold)
        if num_correct != len(gold):
            incorrect_indices.append(idx)
    print(f"Accuracy: {correct / total}, sample size {total}, over {len(golds)} examples")
    return incorrect_indices


def test():
    vocab_size = 10
    examples, repeated, golds = generate_dataset(vocab_size=vocab_size)
    print(f"{examples.shape=}, {repeated.shape=}, {golds.shape=}")
    incorrect_indices = score_solution(repeated, examples, golds)
    print(incorrect_indices)


if __name__ == '__main__':
    test()
