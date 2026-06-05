#!/usr/bin/env python3
# coding: utf-8
"""Load local Korean sample datasets for the `--smoke_test test_cases` path.

This exercises the real data pipeline (dynamic data loader + collators) on a
handful of bundled Korean parquet samples, rather than the synthetic dummy
collator. Used by `get_streaming_datasets` in train_han2han when
`args.smoke_test == 'test_cases'`.
"""

from pathlib import Path

import polars as pl
from datasets import Dataset

from dynamic_data_loader import DataSourceConfig, get_sampling_ratios_from_sources

TEST_DATA_DIR = Path('test_data')

# (filename, source_name, data_type, display_name)
# each file needs a unique source_name to avoid dict overwrites
TEST_CASES = [
    ('finepdfs_korean_sample.parquet', 'finepdfs_korean', 'denoising', 'Fine PDFs Korean (denoising)'),
]


def create_test_source_configs():
    """Create DataSourceConfig objects for the bundled Korean test datasets."""
    return {
        'finepdfs_korean': DataSourceConfig(
            name='finepdfs_korean',
            gcs_pattern='',
            weight=0.10,
            data_type='denoising',
            text_field='text',
            metadata_field='url',
        ),
    }


def load_test_datasets_for_training(n_samples_per_source: int = 100000):
    """Load bundled Korean test datasets for a local training smoke run.

    Args:
        n_samples_per_source: number of samples to load per source file

    Returns:
        tuple: (datasets dict, sampling_ratios dict, source_configs dict)
    """
    datasets = {}
    source_configs = create_test_source_configs()

    for filename, source_name, data_type, _ in TEST_CASES:
        file_path = TEST_DATA_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Test sample missing: {file_path}. The `test_cases` smoke path "
                f"requires the bundled parquet samples under {TEST_DATA_DIR}/."
            )

        df = pl.read_parquet(file_path).head(n_samples_per_source)
        ds = Dataset.from_polars(df)

        ds.info.source_type = source_name
        ds.info.data_type = data_type

        datasets[source_name] = ds

    source_list = [source_configs[name] for name in datasets.keys() if name in source_configs]
    sampling_ratios = get_sampling_ratios_from_sources(source_list)

    return datasets, sampling_ratios, source_configs
