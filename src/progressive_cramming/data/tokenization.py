from __future__ import annotations

import hashlib
import json
import os

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def load_or_create_tokenized_dataset(
    cache_dir: str,
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    model_checkpoint: str,
    no_bos_token: bool = False,
    limit_dataset_items: int | None = None,
    offset_dataset_items: int | None = None,
    cache_prefix: str = "dataset",
    num_proc: int = 4,
    fallback_length: int | None = None,
) -> Dataset:
    """Load a tokenized dataset from `cache_dir` if it exists, otherwise tokenize and persist it."""
    cache_params = {
        "dataset": dataset_name,
        "split": split,
        "limit_dataset_items": limit_dataset_items,
        "offset_dataset_items": offset_dataset_items,
        "max_sequence_length": max_sequence_length,
        "model_checkpoint": model_checkpoint,
        "no_bos_token": no_bos_token,
    }
    cache_key_json = json.dumps(cache_params, sort_keys=True, ensure_ascii=False, default=str)
    cache_key_hash = hashlib.sha256(cache_key_json.encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{cache_prefix}_{cache_key_hash}")

    if os.path.exists(cache_path):
        print(f"Loading tokenized dataset from cache: {cache_path}")
        return Dataset.load_from_disk(cache_path).with_format("torch")

    print("Loading and tokenizing dataset (this may take a while)...")
    raw_dataset = _load_raw_dataset(dataset_name, split, num_proc)
    selected_dataset = _select_range(raw_dataset, offset_dataset_items, limit_dataset_items, fallback_length)
    tokenized_dataset = _tokenize_dataset(selected_dataset, tokenizer, max_sequence_length, no_bos_token, num_proc)

    print(f"Saving tokenized dataset to cache: {cache_path}")
    tokenized_dataset.save_to_disk(cache_path)
    return tokenized_dataset.with_format("torch")


def _load_raw_dataset(dataset_name: str, split: str, num_proc: int) -> Dataset:
    """Load the raw HuggingFace dataset, handling the dataset-specific quirks."""
    if dataset_name in ("mrsndmn/pg19-model-sampled-llama3.1-8B-prefix-64-max_len-2048",):
        split = "train"

    kwargs = {"num_proc": num_proc, "split": split}
    if dataset_name == "HuggingFaceFW/fineweb-edu":
        del kwargs["num_proc"]
        del kwargs["split"]
        kwargs["data_files"] = [f"sample/10BT/{i:03}_00000.parquet" for i in range(14)]
    elif dataset_name in ("LarryLovestein/pg19_1k", "LarryLovestein/fanfics_1k"):
        kwargs["split"] = "train"

    raw_dataset = load_dataset(dataset_name, **kwargs)
    if dataset_name == "HuggingFaceFW/fineweb-edu":
        raw_dataset = raw_dataset["train"]
    return raw_dataset


def _select_range(
    raw_dataset: Dataset,
    offset_dataset_items: int | None,
    limit_dataset_items: int | None,
    fallback_length: int | None,
) -> Dataset:
    """Slice `raw_dataset` according to offset/limit/fallback semantics."""
    if offset_dataset_items is not None:
        start_idx = offset_dataset_items
        if limit_dataset_items is not None:
            return raw_dataset.select(range(start_idx, start_idx + limit_dataset_items))
        if fallback_length is not None:
            return raw_dataset.select(range(start_idx, start_idx + fallback_length))
        return raw_dataset.select(range(start_idx, len(raw_dataset)))
    if limit_dataset_items is not None:
        return raw_dataset.select(range(limit_dataset_items))
    if fallback_length is not None:
        return raw_dataset.select(range(fallback_length))
    return raw_dataset


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    no_bos_token: bool,
    num_proc: int,
) -> Dataset:
    """Tokenize `dataset` with optional BOS suppression; mutates tokenizer state transiently."""
    add_bos_supported = hasattr(tokenizer, "add_bos_token")
    original_add_bos = getattr(tokenizer, "add_bos_token", None)
    if no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = False

    def _tokenize(example):
        # Important: do not pass return_tensors="pt" — HF Datasets stores tensors as
        # nested lists like [1, T] which make __getitem__ very slow. Plain lists +
        # .with_format("torch") on the resulting dataset is the correct pattern.
        add_special_tokens = True
        if no_bos_token and not add_bos_supported:
            add_special_tokens = False
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
            add_special_tokens=add_special_tokens,
        )

    tokenized = dataset.map(_tokenize, num_proc=num_proc, remove_columns=dataset.column_names)
    if no_bos_token and add_bos_supported:
        tokenizer.add_bos_token = original_add_bos
    return tokenized
