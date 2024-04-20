import numpy as np
import torch


def context_query_value(
        vocab_size: int,
        num_examples: int,
        seed: int,
        num_kv_pairs: int = 8):
    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_shuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_shuffled, replace=False, size=num_kv_pairs)

    values_shuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_shuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    return context_size, keys, values, kvs


def multiquery_ar(
        vocab_size: int,
        num_examples: int,
        input_seq_len: int,
        seed: int,
        power_a: float = 0.01,
        num_kv_pairs: int = 8,
        random_non_queries: bool = True
) -> dict:
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    context_size, keys, values, kvs = context_query_value(
        vocab_size=vocab_size,
        num_examples=num_examples,
        seed=seed,
        num_kv_pairs=num_kv_pairs
    )

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs,
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return dict(
        inputs=inputs,
        labels=labels,
        slices=dict(
            num_kv_pairs=num_kv_pairs,
            input_seq_len=input_seq_len,
        )
    )

