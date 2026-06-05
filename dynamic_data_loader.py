#!/usr/bin/env python3
# coding: utf-8
"""
Dynamic data loading system for Han2Han training.
Handles adaptive slicing based on num_hosts and memory constraints.
"""

import os
os.environ['POLARS_MAX_THREADS'] = '4'

import polars as pl
from datasets import load_dataset, Dataset
from typing import Dict, Optional, List, Literal
from etils import epath

import logging

from logging_utils import log_from_main_process, log_from_all_processes

logger = logging.getLogger(__name__)


def stratified_sample(
    df: pl.DataFrame,
    target_size: int,
    group_col: str,
    seed: int,
) -> pl.DataFrame:
    """Sample proportionally from each group in a DataFrame (stratified sampling).

    This ensures each group/category in the data is represented proportionally
    in the sampled output, rather than being randomly over/under-represented.

    Args:
        df: Polars DataFrame to sample from
        target_size: Target number of rows to sample
        group_col: Column name to group by for stratification
        seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame with proportional representation from each group
    """
    if group_col not in df.columns:
        return df.sample(n=min(target_size, len(df)), seed=seed)

    if len(df) <= target_size:
        return df

    scale_factor = target_size / len(df)
    unique_groups = df[group_col].unique().to_list()
    sampled_dfs = []

    for group in unique_groups:
        group_df = df.filter(pl.col(group_col) == group)
        group_count = len(group_df)
        sample_size = max(1, int(group_count * scale_factor))
        sampled_dfs.append(group_df.sample(n=min(sample_size, group_count), seed=seed))

    return pl.concat(sampled_dfs) if sampled_dfs else df


class DataSourceConfig:
    """Configuration for a single data source with field mappings."""

    def __init__(
        self,
        name: str,
        gcs_pattern: str,
        weight: float,
        data_type: Literal['denoising', 'ocr_correction', 'temporal_classification', 'temporal_continuation', 'sts', 'transcription', 'mixed', 'topic_classification', 'nli', 'instruction_following', 'multiple_choice', 'summarization', 'cot_reasoning'],
        text_field: str = 'text',
        metadata_field: str = 'url',
        target_field: Optional[str] = None,
        year_field: Optional[str] = None,
        sentence1_field: Optional[str] = None,
        sentence2_field: Optional[str] = None,
        score_field: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
        eval_pattern: Optional[str] = None,
        test_pattern: Optional[str] = None,
        has_stratified_split: bool = False,
    ):
        """
        Args:
            name: unique identifier for this source
            gcs_pattern: GCS path pattern (supports wildcards like gs://bucket/*.parquet)
            weight: proportion of total training budget (should sum to 1.0 across all sources)
            data_type: type of task for this data
            text_field: name of the main text/input column
            metadata_field: name of the metadata column (url, title, etc.)
            target_field: name of target column for translation/transcription/OCR
            year_field: name of year/date column for temporal classification
            sentence1_field: first sentence for STS tasks
            sentence2_field: second sentence for STS tasks
            score_field: similarity score for STS tasks
            required_columns: list of required columns (auto-detected if None)
            eval_pattern: GCS path pattern for validation data (if pre-split exists)
            test_pattern: GCS path pattern for test data (for final evaluation)
            has_stratified_split: whether this source has proper stratified train/eval/test splits
        """
        self.name = name
        self.gcs_pattern = gcs_pattern
        self.weight = weight
        self.data_type = data_type
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.target_field = target_field
        self.year_field = year_field
        self.sentence1_field = sentence1_field
        self.sentence2_field = sentence2_field
        self.score_field = score_field
        self.eval_pattern = eval_pattern
        self.test_pattern = test_pattern
        self.has_stratified_split = has_stratified_split

        # auto-detect required columns if not specified
        if required_columns is None:
            required_columns = [text_field]
            if metadata_field and metadata_field not in required_columns:
                required_columns.append(metadata_field)
            if target_field and target_field != text_field:  # avoid duplicates
                required_columns.append(target_field)
            if year_field and year_field not in required_columns:
                required_columns.append(year_field)
            if sentence1_field and sentence1_field not in required_columns:
                required_columns.append(sentence1_field)
            if sentence2_field and sentence2_field not in required_columns:
                required_columns.append(sentence2_field)
            if score_field and score_field not in required_columns:
                required_columns.append(score_field)

        # deduplicate just in case
        self.required_columns = list(dict.fromkeys(required_columns))

    def __repr__(self):
        return f"DataSourceConfig(name={self.name}, type={self.data_type}, weight={self.weight})"


class DynamicDataLoader:
    """Adaptive data loader that respects memory and host constraints."""

    def __init__(
        self,
        cache_dir: str = None,  # derived from data_bucket if None
        max_disk_gb_per_host: float = 30.0,
        compression: str = 'zstd',
        compression_level: int = 9,
        disable_budget_limit: bool = False,
        data_bucket: str = "gs://r4z0kd9han2han",
    ):
        """
        Args:
            cache_dir: cache directory (derived from data_bucket if None)
            max_disk_gb_per_host: maximum disk space per host (safety limit, ignored if disable_budget_limit=True)
            compression: parquet compression codec
            compression_level: compression level (1-22 for zstd, higher = better compression)
            disable_budget_limit: if True, use full data slice without truncating (assumes you know it fits)
            data_bucket: GCS bucket prefix for training data sources
        """
        # derive cache_dir from data_bucket if not explicitly provided
        if cache_dir is None:
            cache_dir = f"{data_bucket}/pt_data_cache"
        # use epath.Path for unified GCS/local path handling
        self.cache_dir = epath.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_disk_gb_per_host = max_disk_gb_per_host
        self.compression = compression
        self.compression_level = compression_level
        self.disable_budget_limit = disable_budget_limit
        self.data_bucket = data_bucket

    def calculate_budget(
        self,
        num_hosts: int,
        training_mode: Literal['full', 'sweep', 'debug'] = 'full',
    ) -> float:
        """Calculate per-host data budget in GB.

        Args:
            num_hosts: number of training hosts
            training_mode: 'full' (300B tokens), 'sweep' (~10B), 'debug' (~1B)

        Returns:
            budget_gb: gigabytes of data this host should load
        """
        # total data budgets by mode
        mode_budgets = {
            'full': 320.0,  # all data (~300B tokens, ~320 GB actual)
            'sweep': 10.0,  # ~10B tokens (3% of full)
            'debug': 1.0,   # ~1B tokens for quick testing
        }

        total_budget_gb = mode_budgets[training_mode]

        # fair split across hosts, capped at safety limit
        per_host_gb = min(
            total_budget_gb / num_hosts,
            self.max_disk_gb_per_host
        )

        log_from_main_process(logger, 'info', f"Budget calculation: mode={training_mode}, hosts={num_hosts}, per_host={per_host_gb:.2f}GB")
        return per_host_gb


    def load_source_slice(
        self,
        source_config: DataSourceConfig,
        host_idx: int,
        num_hosts: int,
        budget_gb: float,
        training_mode: Literal['full', 'sweep', 'debug'] = 'full',
        force_reload: bool = False,
    ) -> Dataset:
        """Load non-overlapping slice of a data source.

        Args:
            source_config: configuration for the data source
            host_idx: index of this host (0-based)
            num_hosts: total number of hosts
            budget_gb: allocated budget for this source in GB
            training_mode: training mode (included in cache key for proper isolation)
            force_reload: if True, ignore cache and reload from GCS

        Returns:
            datasets.Dataset with this host's slice
        """
        # check for cached parquet first (include training_mode and per-source filter values in cache key)
        cache_path = self.cache_dir / f"{source_config.name}{training_mode}_host{host_idx}_of{num_hosts}.parquet"
        if cache_path.exists() and not force_reload:
            log_from_all_processes(logger, 'info', f"Loading {source_config.name} from cache: {cache_path}")
            return load_dataset('parquet', data_files=str(cache_path), split='train')

        # use native gs:// paths for Polars scanning (faster than gcsfuse!)
        log_from_all_processes(logger, 'info', f"Scanning {source_config.gcs_pattern} for {source_config.name}")

        # Special handling for han2han_curated: stratified slicing by source
        if source_config.name == 'han2han_curated':
            # Need to include 'source' for stratified slicing
            required_cols = list(set(source_config.required_columns + ['source']))
            df_lazy = pl.scan_parquet(source_config.gcs_pattern).select(required_cols)

            # Get unique sources and total count
            source_info = df_lazy.group_by('source').agg(pl.len().alias('count')).collect()

            # Collect slices from each source proportionally
            dfs_to_concat = []
            for row in source_info.iter_rows(named=True):
                source_name = row['source']
                source_total = row['count']

                # Calculate this host's slice of this source
                rows_per_host = source_total // num_hosts
                start_row = host_idx * rows_per_host

                if host_idx == num_hosts - 1:
                    # Last host gets remainder
                    rows_to_take = source_total - start_row
                else:
                    rows_to_take = rows_per_host

                # Slice this source's data
                source_slice = (
                    df_lazy
                    .filter(pl.col('source') == source_name)
                    .slice(start_row, rows_to_take)
                    .collect()
                )
                dfs_to_concat.append(source_slice)

            df = pl.concat(dfs_to_concat)
            log_from_all_processes(logger, 'info',
                f"{source_config.name}: collected stratified slices from {len(source_info)} sources")
        else:
            # Regular slicing for other datasets
            df_lazy = pl.scan_parquet(source_config.gcs_pattern).select(source_config.required_columns)

            # calculate non-overlapping slice boundaries
            total_rows = df_lazy.select(pl.count()).collect().item()
            rows_per_host = total_rows // num_hosts
            start_row = host_idx * rows_per_host

            # last host gets remainder
            if host_idx == num_hosts - 1:
                end_row = total_rows
            else:
                end_row = start_row + rows_per_host

            log_from_all_processes(logger, 'info', f"{source_config.name}: total_rows={total_rows}, host {host_idx} gets rows {start_row}-{end_row}")

            # slice and collect - this only downloads the data we need!
            # Since we're in same zone, bandwidth is free and fast
            df = df_lazy.slice(start_row, end_row - start_row).collect()

        # estimate size and truncate if over budget (unless budget limit disabled)
        estimated_gb = df.estimated_size() / (1024**3)
        log_from_all_processes(logger, 'info', f"{source_config.name}: estimated_size={estimated_gb:.2f}GB, budget={budget_gb:.2f}GB")

        if not self.disable_budget_limit and estimated_gb > budget_gb:
            scale_factor = budget_gb / estimated_gb
            keep_rows = int(len(df) * scale_factor)
            log_from_all_processes(logger, 'warning', f"{source_config.name}: truncating {len(df)} -> {keep_rows} rows to fit budget")

            # deterministic seed based on host and dataset
            seed = hash((source_config.name, host_idx, num_hosts)) % (2**32)

            # stratified sampling for sources with 'source' column (e.g. han2han_curated)
            if source_config.name == 'han2han_curated' and 'source' in df.columns:
                try:
                    df = stratified_sample(df, target_size=keep_rows, group_col='source', seed=seed)
                    log_from_all_processes(logger, 'info', f"han2han_curated: stratified sampling complete")
                except Exception as e:
                    log_from_all_processes(logger, 'warning', f"han2han_curated stratified sampling failed: {e}, falling back to random")
                    df = df.sample(n=keep_rows, seed=seed)
            else:
                df = df.sample(n=keep_rows, seed=seed)

        # cache as compressed parquet before converting to Dataset (saves memory!)
        cache_path = self.cache_dir / f"{source_config.name}{filter_suffix}_{training_mode}_host{host_idx}_of{num_hosts}.parquet"
        log_from_all_processes(logger, 'info', f"Saving compressed parquet to cache: {cache_path}")
        df.write_parquet(
            str(cache_path),  # convert epath to str for Polars
            compression=self.compression,
            compression_level=self.compression_level,
        )

        # convert to Dataset and return
        log_from_all_processes(logger, 'info', f"Converting to Dataset from Polars")
        return Dataset.from_polars(df)

    def load_all_sources(
        self,
        sources: List[DataSourceConfig],
        host_idx: int,
        num_hosts: int,
        training_mode: Literal['full', 'sweep', 'debug'] = 'full',
        force_reload: bool = False,
    ) -> Dict[str, Dataset]:
        """Load slices for all configured data sources.

        Args:
            sources: list of data source configurations
            host_idx: index of this host
            num_hosts: total number of hosts
            training_mode: training mode for budget calculation
            force_reload: if True, ignore cache and reload all sources

        Returns:
            dict mapping source names to datasets
        """
        # calculate total budget
        total_budget_gb = self.calculate_budget(num_hosts, training_mode)

        # normalize weights
        total_weight = sum(s.weight for s in sources)
        if abs(total_weight - 1.0) > 0.01:
            log_from_all_processes(logger, 'warning', f"Source weights sum to {total_weight}, normalizing to 1.0")

        # load each source
        loaded_sources = {}
        for source in sources:
            # allocate budget proportionally
            source_budget_gb = total_budget_gb * (source.weight / total_weight)

            log_from_all_processes(logger, 'info', f"Loading {source.name} (budget={source_budget_gb:.2f}GB)")
            dataset = self.load_source_slice(
                source,
                host_idx,
                num_hosts,
                source_budget_gb,
                training_mode,
                force_reload
            )

            loaded_sources[source.name] = dataset
            log_from_all_processes(logger, 'info', f"Loaded {source.name}: {len(dataset)} examples")

        return loaded_sources

    def load_sft_source_slice(
        self,
        source_config: DataSourceConfig,
        host_idx: int,
        num_hosts: int,
        split: str,
        force_reload: bool = False,
    ) -> tuple[Dataset, int]:
        """Load a non-overlapping per-host slice of a single SFT parquet source.

        Mirrors load_source_slice() but specialized for SFT: no budget logic,
        no stratified branch, split-aware cache key, and the returned dataset
        is the memory-mapped Arrow view produced by datasets.load_dataset.
        A sidecar text file preserves the pre-slice row count so cache hits
        can still feed the FLAN-style sampling-ratio math.

        Args:
            source_config: SFT source configuration (gcs_pattern points at a
                single parquet, either local or gs://).
            host_idx: index of this host (0-based).
            num_hosts: total number of hosts (use 1 for single-host runs).
            split: split name ('train', 'validation', 'test') included in the
                cache key so splits never share shards.
            force_reload: if True, ignore any existing cache entry.

        Returns:
            Tuple of (sliced memory-mapped Dataset, full unsliced row count).
        """
        cache_path = self.cache_dir / f"sft_{source_config.name}_{split}_host{host_idx}_of{num_hosts}.parquet"
        sidecar_path = self.cache_dir / f"sft_{source_config.name}_{split}_fullsize.txt"

        if cache_path.exists() and sidecar_path.exists() and not force_reload:
            full_size = int(sidecar_path.read_text().strip())
            log_from_all_processes(logger, 'info',
                f"Loading SFT {source_config.name} ({split}) from cache: {cache_path} "
                f"(full_size={full_size:,})")
            ds = load_dataset('parquet', data_files=str(cache_path), split='train')
            return ds, full_size

        # pre-check existence so a missing split (e.g. NLI test) surfaces as
        # FileNotFoundError -- get_local_sft_datasets relies on that to skip
        # optional splits without aborting the whole run.
        source_path = epath.Path(source_config.gcs_pattern)
        if not source_path.exists():
            raise FileNotFoundError(
                f"SFT parquet not found for {source_config.name} ({split}): "
                f"{source_config.gcs_pattern}"
            )

        log_from_all_processes(logger, 'info',
            f"Scanning {source_config.gcs_pattern} for SFT {source_config.name} ({split})")

        try:
            df_lazy = pl.scan_parquet(source_config.gcs_pattern)
            total_rows = df_lazy.select(pl.count()).collect().item()
        except Exception as e:
            raise RuntimeError(
                f"Failed to scan SFT source {source_config.name} ({split}) "
                f"at {source_config.gcs_pattern}: {e}"
            ) from e

        rows_per_host = total_rows // num_hosts
        start_row = host_idx * rows_per_host
        end_row = total_rows if host_idx == num_hosts - 1 else start_row + rows_per_host

        log_from_all_processes(logger, 'info',
            f"SFT {source_config.name} ({split}): total_rows={total_rows:,}, "
            f"host {host_idx}/{num_hosts} gets rows {start_row:,}-{end_row:,}")

        df = df_lazy.slice(start_row, end_row - start_row).collect()

        log_from_all_processes(logger, 'info',
            f"Saving SFT cache shard to {cache_path} ({len(df):,} rows)")
        df.write_parquet(
            str(cache_path),
            compression=self.compression,
            compression_level=self.compression_level,
        )
        sidecar_path.write_text(str(total_rows))

        del df

        ds = load_dataset('parquet', data_files=str(cache_path), split='train')
        return ds, total_rows


def create_han2han_data_sources(
    data_bucket: str = "gs://r4z0kd9han2han",
    sft_tasks: str = "all",
) -> List[DataSourceConfig]:
    """Define Korean-only data sources for Han2Han training.

    Han2Han focuses on Korean language understanding with Hanja-Hangul mapping.
    Sources are filtered to Korean-only data.

    Args:
        data_bucket: GCS bucket prefix for training data
        sft_tasks: Which SFT tasks to include. Options:
            - 'all': include all sources (default)
            - 'none': only pretraining data (denoising/mixed)
            - comma-separated list: specific task types (e.g., 'sts,nli,ynat')

    Returns:
        list of Korean-only data sources (weights sum to ~1.0)
    """
    sources = [
        # === KOREAN PRETRAINING (70% of budget) ===
        DataSourceConfig(
            name='han2han_curated',
            gcs_pattern=f'{data_bucket}/multilingual_data/cadazar_cjd-metadata-morphs_default_train_shard*.parquet',
            weight=0.35,
            data_type='mixed',
            text_field='original_text',
            metadata_field='metadata',
        ),
        DataSourceConfig(
            name='mc4_ko',
            gcs_pattern=f'{data_bucket}/multilingual_data/allenai_c4_ko_train_shard*.parquet',
            weight=0.02,
            data_type='denoising',
            text_field='text',
            metadata_field='url',
        ),
        DataSourceConfig(
            name='finepdfs_korean',
            gcs_pattern=f'{data_bucket}/pretraining_data/finepdfs_korean/*.parquet',
            weight=0.35,
            data_type='denoising',
            text_field='text',
            metadata_field='url',
        ),

        # === KOREAN TASK DATA (30% of budget) ===
        DataSourceConfig(
            name='korean_historical',
            gcs_pattern=f'{data_bucket}/task_data/korean_historical/*.parquet',
            weight=0.15,
            data_type='mixed',
            text_field='text',
            metadata_field='source',
            year_field='year',
        ),

        DataSourceConfig(
            name='kornlu_sts',
            gcs_pattern=f'{data_bucket}/task_data/sts/train.parquet',
            weight=0.00333,
            data_type='sts',
            text_field='input_text',
            metadata_field='input_text',
            sentence1_field='sentence1',
            sentence2_field='sentence2',
            score_field='rounded_score',
            eval_pattern=f'{data_bucket}/task_data/sts/validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),

        # KLUE YNAT (topic classification)
        DataSourceConfig(
            name='klue_ynat',
            gcs_pattern=f'{data_bucket}/task_data/ynat/*_train.parquet',
            weight=0.00333,
            data_type='topic_classification',
            text_field='input_text',
            target_field='target_text',
            metadata_field='task',
            eval_pattern=f'{data_bucket}/task_data/ynat/*_validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),

        # KLUE NLI (natural language inference)
        DataSourceConfig(
            name='klue_nli',
            gcs_pattern=f'{data_bucket}/task_data/nli/*_train.parquet',
            weight=0.00333,
            data_type='nli',
            text_field='input_text',
            target_field='target_text',
            metadata_field='task',
            eval_pattern=f'{data_bucket}/task_data/nli/*_validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
    ]

    # filter sources based on sft_tasks parameter
    pretraining_types = {'denoising', 'mixed', 'temporal_continuation'}
    sft_tasks_lower = sft_tasks.lower().strip() if sft_tasks else 'all'
    log_from_all_processes(logger, 'debug', f"sft_tasks filter: '{sft_tasks}' -> '{sft_tasks_lower}'")
    log_from_all_processes(logger, 'debug', f"Sources before filter: {[s.name for s in sources]}")

    if sft_tasks_lower == 'all':
        pass
    elif sft_tasks_lower == 'none':
        sources = [s for s in sources if s.data_type in pretraining_types]
    else:
        allowed_sft = set(t.strip() for t in sft_tasks_lower.split(','))
        sources = [
            s for s in sources
            if s.data_type in pretraining_types or s.data_type in allowed_sft
        ]

    log_from_all_processes(logger, 'info', f"Sources after sft_tasks filter: {[s.name for s in sources]}")

    return sources


def get_sft_eval_sources(
    sft_tasks: str = "all",
    data_bucket: str = "gs://r4z0kd9han2han",
    local: bool = False,
    local_base_dir: str = "task_data",
) -> List[DataSourceConfig]:
    """Return DataSourceConfigs for SFT tasks only (for generative evaluation).

    This returns only the supervised fine-tuning task configs, without any
    pretraining data. Used to create a separate eval collator for generative
    evaluation of classification/regression tasks.

    Args:
        sft_tasks: Which SFT tasks to include ('all', 'none', or comma-separated list
                   like 'nli,sts,topic_classification,temporal_classification')
        data_bucket: GCS bucket prefix for data (ignored if local=True)
        local: If True, use local paths instead of GCS
        local_base_dir: Base directory for local data (only used if local=True)

    Returns:
        List of DataSourceConfig for SFT tasks only
    """
    if local:
        base = local_base_dir
        sts_pattern = f'{base}/sts/validation.parquet'
        ynat_pattern = f'{base}/ynat/validation.parquet'
        nli_pattern = f'{base}/nli/validation.parquet'
        temporal_pattern = f'{base}/temporal_classification/validation.parquet'
    else:
        sts_pattern = f'{data_bucket}/task_data/sts/validation.parquet'
        ynat_pattern = f'{data_bucket}/task_data/ynat/*_validation.parquet'
        nli_pattern = f'{data_bucket}/task_data/nli/*_validation.parquet'
        temporal_pattern = f'{data_bucket}/task_data/temporal_classification/korean_temporal_validation.parquet'

    sft_sources = [
        DataSourceConfig(
            name='kornlu_sts',
            gcs_pattern=sts_pattern,
            weight=1.0,
            data_type='sts',
            text_field='input_text',
            metadata_field='input_text',
            target_field='target_text',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='klue_ynat',
            gcs_pattern=ynat_pattern,
            weight=1.0,
            data_type='topic_classification',
            text_field='input_text',
            target_field='target_text',
            metadata_field='task',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='klue_nli',
            gcs_pattern=nli_pattern,
            weight=1.0,
            data_type='nli',
            text_field='input_text',
            target_field='target_text',
            metadata_field='task',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='korean_temporal',
            gcs_pattern=temporal_pattern,
            weight=1.0,
            data_type='temporal_classification',
            text_field='text',
            metadata_field='source',
            year_field='year',
            has_stratified_split=True,
        ),
    ]

    sft_tasks_lower = sft_tasks.lower().strip() if sft_tasks else 'all'

    if sft_tasks_lower == 'none':
        return []
    elif sft_tasks_lower != 'all':
        allowed = set(t.strip() for t in sft_tasks_lower.split(','))
        sft_sources = [s for s in sft_sources if s.data_type in allowed]

    mode = "local" if local else "GCS"
    log_from_all_processes(logger, 'info',
        f"SFT eval sources ({mode}): {[s.name for s in sft_sources]}")
    return sft_sources


def get_sft_eval_datasets(
    sft_tasks: str = "all",
    data_bucket: str = "gs://r4z0kd9han2han",
    host_idx: int = 0,
    num_hosts: int = 1,
) -> tuple[Dict[str, Dataset], Dict[str, float], Dict[str, DataSourceConfig]]:
    """Load SFT evaluation datasets for generative evaluation callback.

    This loads only the SFT task datasets (not pretraining data) for use with
    the GenerativeEvaluationCallback. The data is loaded with minimal processing
    to support small eval samples.

    Args:
        sft_tasks: Which SFT tasks to include ('all', 'none', or comma-separated
                   list like 'nli,sts,topic_classification,temporal_classification')
        data_bucket: GCS bucket prefix for data
        host_idx: index of this host (for distributed loading)
        num_hosts: total number of hosts

    Returns:
        tuple of (datasets dict, sampling_ratios dict, source_configs dict)
    """
    sources = get_sft_eval_sources(sft_tasks=sft_tasks, data_bucket=data_bucket)

    if not sources:
        log_from_all_processes(logger, 'info', "No SFT eval sources configured")
        return {}, {}, {}

    loader = DynamicDataLoader(
        disable_budget_limit=True,
        data_bucket=data_bucket,
    )

    datasets = loader.load_all_sources(
        sources=sources,
        host_idx=host_idx,
        num_hosts=num_hosts,
        training_mode='debug',
        force_reload=False,
    )

    for source in sources:
        if source.name in datasets:
            ds = datasets[source.name]
            ds.info.source_type = source.name
            ds.info.data_type = source.data_type

            log_from_all_processes(logger, 'info',
                f"Loaded SFT eval {source.name}: {len(ds):,} examples "
                f"(type={source.data_type})"
            )

    sampling_ratios = get_sampling_ratios_from_sources(sources)
    source_configs = {source.name: source for source in sources}

    return datasets, sampling_ratios, source_configs


def get_eval_sources_from_configs(
    train_sources: List[DataSourceConfig],
) -> List[DataSourceConfig]:
    """Extract eval sources from train configs that have eval_pattern set.

    This creates new DataSourceConfig objects for validation data from
    train sources that have pre-defined stratified splits.

    Args:
        train_sources: List of training DataSourceConfigs

    Returns:
        List of DataSourceConfig for eval data (only sources with eval_pattern)
    """
    eval_sources = []
    for src in train_sources:
        if src.eval_pattern:
            eval_src = DataSourceConfig(
                name=f"{src.name}_eval",
                gcs_pattern=src.eval_pattern,
                weight=src.weight,
                data_type=src.data_type,
                text_field=src.text_field,
                metadata_field=src.metadata_field,
                target_field=src.target_field,
                year_field=src.year_field,
                sentence1_field=src.sentence1_field,
                sentence2_field=src.sentence2_field,
                score_field=src.score_field,
                required_columns=src.required_columns,
                eval_pattern=None,
                test_pattern=None,
                has_stratified_split=True,
            )
            eval_sources.append(eval_src)
    return eval_sources


def get_test_sources_from_configs(
    train_sources: List[DataSourceConfig],
) -> List[DataSourceConfig]:
    """Extract test sources from train configs that have test_pattern set.

    This creates new DataSourceConfig objects for test data from
    train sources that have pre-defined stratified splits.

    Args:
        train_sources: List of training DataSourceConfigs

    Returns:
        List of DataSourceConfig for test data (only sources with test_pattern)
    """
    test_sources = []
    for src in train_sources:
        if src.test_pattern:
            test_src = DataSourceConfig(
                name=f"{src.name}_test",
                gcs_pattern=src.test_pattern,
                weight=src.weight,
                data_type=src.data_type,
                text_field=src.text_field,
                metadata_field=src.metadata_field,
                target_field=src.target_field,
                year_field=src.year_field,
                sentence1_field=src.sentence1_field,
                sentence2_field=src.sentence2_field,
                score_field=src.score_field,
                required_columns=src.required_columns,
                eval_pattern=None,
                test_pattern=None,
                has_stratified_split=True,
            )
            test_sources.append(test_src)
    return test_sources


def get_han2han_datasets(
    host_idx: int = 0,
    num_hosts: int = 8,
    training_mode: Literal['full', 'sweep', 'debug'] = 'full',
    force_reload: bool = False,
    disable_budget_limit: bool = False,
    data_bucket: str = "gs://r4z0kd9han2han",
    sft_tasks: str = "all",
) -> tuple[Dict[str, Dataset], Dict[str, float], Dict[str, DataSourceConfig]]:
    """
    Load Han2Han (Korean-only) datasets with proper metadata for training.

    Used for training the Han2Han base model focused on Korean language
    understanding.

    Args:
        host_idx: index of this host (0-based)
        num_hosts: total number of training hosts
        training_mode: 'full' (150B tokens), 'sweep' (~10B), or 'debug' (~1B)
        force_reload: if True, ignore cache and reload from GCS
        disable_budget_limit: if True, use full data slices without budget truncation
        data_bucket: GCS bucket prefix for training data
        sft_tasks: Which SFT tasks to include ('all', 'none', or comma-separated list)

    Returns:
        tuple of (datasets dict, sampling_ratios dict, source_configs dict)
    """
    loader = DynamicDataLoader(
        disable_budget_limit=disable_budget_limit,
        data_bucket=data_bucket,
    )

    sources = create_han2han_data_sources(
        data_bucket=data_bucket, sft_tasks=sft_tasks,
    )

    datasets = loader.load_all_sources(
        sources=sources,
        host_idx=host_idx,
        num_hosts=num_hosts,
        training_mode=training_mode,
        force_reload=force_reload,
    )

    for source in sources:
        if source.name in datasets:
            ds = datasets[source.name]
            ds.info.source_type = source.name
            ds.info.data_type = source.data_type

            log_from_all_processes(logger, 'info',
                f"Loaded {source.name}: {len(ds):,} examples "
                f"(type={source.data_type}"
            )

    sampling_ratios = get_sampling_ratios_from_sources(sources)
    source_configs = {source.name: source for source in sources}

    return datasets, sampling_ratios, source_configs


def get_sampling_ratios_from_sources(
    sources: List[DataSourceConfig],
    korean_hanja_heavy_ratio: float = 0.18
) -> Dict[str, float]:
    """
    Convert source weights to sampling ratios for the collator.

    Special handling for Korean DENOISING and MIXED sources: splits them into
    'korean_hanja_heavy' and 'korean_hanja_light' sub-sources based on hanja_heavy_ratio.
    This matches the collator's get_next_balanced_sample() which routes both denoising
    and mixed Korean data through hanja classification.

    Korean supervised tasks (STS, translation, transcription, temporal) keep their source names.

    Args:
        sources: List of data source configurations with weight attributes
        korean_hanja_heavy_ratio: Fraction of Korean denoising samples that are hanja-heavy (default: 0.18 = 18%)

    Returns:
        Dict mapping source names to sampling ratios (sums to 1.0)
        - Korean DENOISING/MIXED sources become 'korean_hanja_heavy' and 'korean_hanja_light'
        - All other sources keep their dataset names
    """
    total_weight = sum(s.weight for s in sources)

    # separate Korean denoising vs everything else
    korean_denoising_weight = 0.0
    other_ratios = {}

    for source in sources:
        ratio = source.weight / total_weight

        # split Korean DENOISING/MIXED data into hanja_heavy/hanja_light
        # supervised tasks (sts, translation, transcription, temporal) use direct sampling
        needs_hanja_split = source.data_type in ('denoising', 'mixed')

        if needs_hanja_split:
            korean_denoising_weight += ratio

    # split Korean denoising weight into heavy/light
    sampling_ratios = {}
    if korean_denoising_weight > 0:
        sampling_ratios['korean_hanja_heavy'] = korean_denoising_weight * korean_hanja_heavy_ratio
        sampling_ratios['korean_hanja_light'] = korean_denoising_weight * (1 - korean_hanja_heavy_ratio)

    # add all other sources
    sampling_ratios.update(other_ratios)

    return sampling_ratios


def create_local_sft_sources(
    base_dir: str = "task_data",
    sft_tasks: str = "all",
    split: str = "train",
) -> List[DataSourceConfig]:
    """Create DataSourceConfigs for local SFT data (no GCS).

    This is for local fine-tuning when data is already prepared locally
    in the task_data/ directory structure.

    Directory structure expected:
        task_data/
            temporal_classification/{train,validation,test}.parquet
            sts/{train,validation,test}.parquet
            nli/{train,validation}.parquet
            ynat/{train,validation}.parquet

    Args:
        base_dir: Base directory containing task subdirectories
        sft_tasks: Which SFT tasks to include ('all' or comma-separated list
                   like 'nli,sts,topic_classification,temporal_classification')
        split: Data split to use ('train', 'validation', 'test')

    Returns:
        List of DataSourceConfig for local SFT tasks
    """
    sources = [
        DataSourceConfig(
            name='korean_temporal',
            gcs_pattern=f'{base_dir}/temporal_classification/korean_temporal_{split}.parquet',
            weight=0.05,
            data_type='temporal_classification',
            text_field='text',
            metadata_field='source',
            target_field=None,
            year_field='year',
            eval_pattern=f'{base_dir}/temporal_classification/korean_temporal_validation.parquet',
            test_pattern=f'{base_dir}/temporal_classification/korean_temporal_test.parquet',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='kornlu_sts',
            gcs_pattern=f'{base_dir}/sts/{split}.parquet',
            weight=0.05,
            data_type='sts',
            text_field='input_text',
            metadata_field='input_text',
            target_field='target_text',
            sentence1_field='sentence1',
            sentence2_field='sentence2',
            score_field='rounded_score',
            eval_pattern=f'{base_dir}/sts/validation.parquet',
            test_pattern=f'{base_dir}/sts/test.parquet',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='klue_nli',
            gcs_pattern=f'{base_dir}/nli/nli_{split}.parquet',
            weight=0.05,
            data_type='nli',
            text_field='input_text',
            target_field='target_text',
            metadata_field='task',
            eval_pattern=f'{base_dir}/nli/nli_validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='klue_ynat',
            gcs_pattern=f'{base_dir}/ynat/ynat_{split}.parquet',
            weight=0.05,
            data_type='topic_classification',
            text_field='input_text',
            target_field='target_text',
            metadata_field='task',
            eval_pattern=f'{base_dir}/ynat/ynat_validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='transcription_articles',
            gcs_pattern=f'{base_dir}/transcription/{split}.parquet',
            weight=0.15,
            data_type='transcription',
            text_field='target',
            metadata_field='target',
            target_field=None,
            eval_pattern=f'{base_dir}/transcription/validation.parquet',
            test_pattern=f'{base_dir}/transcription/test.parquet',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='transcription_sentences',
            gcs_pattern=f'{base_dir}/transcription_sentences/{split}.parquet',
            weight=0.25,
            data_type='transcription',
            text_field='target',
            metadata_field='target',
            target_field=None,
            eval_pattern=f'{base_dir}/transcription_sentences/validation.parquet',
            test_pattern=f'{base_dir}/transcription_sentences/test.parquet',
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='open_kor_instructions',
            gcs_pattern=f'{base_dir}/instruction_following/{split}.parquet',
            weight=0.40,
            data_type='instruction_following',
            text_field='input_text',
            target_field='target_text',
            metadata_field='instruction',
            eval_pattern=f'{base_dir}/instruction_following/validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='komagpie',
            gcs_pattern=f'{base_dir}/komagpie/{split}.parquet',
            weight=0.40,
            data_type='instruction_following',
            text_field='input_text',
            target_field='target_text',
            metadata_field='instruction',
            eval_pattern=f'{base_dir}/komagpie/validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='aihub_summarization',
            gcs_pattern=f'{base_dir}/summarization/{split}.parquet',
            weight=0.20,
            data_type='summarization',
            text_field='text',
            target_field='summary',
            metadata_field='source',
            eval_pattern=f'{base_dir}/summarization/validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
        DataSourceConfig(
            name='kaist_cot_ko',
            gcs_pattern=f'{base_dir}/cot_reasoning/{split}.parquet',
            weight=0.15,
            data_type='cot_reasoning',
            text_field='input_text',
            target_field='target_text',
            metadata_field='kaist_task',
            eval_pattern=f'{base_dir}/cot_reasoning/validation.parquet',
            test_pattern=None,
            has_stratified_split=True,
        ),
    ]

    sft_tasks_lower = sft_tasks.lower().strip() if sft_tasks else 'all'

    if sft_tasks_lower == 'all':
        pass
    else:
        allowed = set(t.strip() for t in sft_tasks_lower.split(','))
        sources = [s for s in sources if s.data_type in allowed or s.name in allowed]

    log_from_all_processes(logger, 'info',
        f"Local SFT sources ({split}): {[s.name for s in sources]}")

    return sources


def load_parquet_dataset(path: str) -> Dataset:
    """Load a parquet file as a HuggingFace Dataset.

    Supports both local paths and GCS paths (gs://...).

    Args:
        path: Local path or GCS URI to parquet file

    Returns:
        Dataset object
    """
    is_gcs = path.startswith('gs://')
    if not is_gcs and not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pl.read_parquet(path)
    return Dataset.from_polars(df)


MC_EVAL_REQUIRED_FIELDS = {'encoder_input', 'candidates', 'correct_idx', 'category', 'benchmark'}


def get_mc_eval_data(
    benchmarks: List[str],
    data_bucket: str,
    max_samples_per_benchmark: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Load pre-formatted MC evaluation benchmarks from GCS.

    Expects parquet files at {data_bucket}/eval/mc/{benchmark}.parquet
    with columns: encoder_input, candidates, correct_idx, category, benchmark.
    Produced by prepare_mc_eval_data.py.

    Args:
        benchmarks: list of benchmark names (e.g., ['kmmlu', 'click', 'haerae'])
        data_bucket: GCS bucket prefix (e.g., gs://r4z0kd9han2han)
        max_samples_per_benchmark: max examples per benchmark (deterministic sampling)
        seed: random seed for sampling

    Returns:
        dict mapping benchmark name -> list of normalized MC examples
    """
    import random

    mc_eval_data = {}

    for benchmark in benchmarks:
        if data_bucket is not None:
            parquet_path = f"{data_bucket.rstrip('/')}/eval/mc/{benchmark}.parquet"
        else:
            parquet_path = f"eval/mc/{benchmark}.parquet"
        try:
            df = pl.read_parquet(parquet_path)
        except Exception as e:
            logger.error(f"Failed to load MC eval {benchmark} from {parquet_path}: {e}")
            continue

        examples = df.to_dicts()
        if not examples:
            logger.warning(f"Empty MC eval data for {benchmark} at {parquet_path}")
            continue

        missing = MC_EVAL_REQUIRED_FIELDS - set(examples[0].keys())
        if missing:
            raise ValueError(
                f"MC eval parquet '{parquet_path}' missing required fields: {missing}. "
                f"Available: {list(examples[0].keys())}. "
                f"Re-run prepare_mc_eval_data.py to regenerate."
            )

        if len(examples) > max_samples_per_benchmark:
            rng = random.Random(seed)
            examples = rng.sample(examples, max_samples_per_benchmark)

        mc_eval_data[benchmark] = examples
        logger.info(f"MC eval: loaded {len(examples)} examples for {benchmark} from {parquet_path}")

    return mc_eval_data


TEMPORAL_EVAL_REQUIRED_FIELDS = {
    'encoder_input', 'candidates', 'correct_idx',
    'category', 'benchmark', 'true_year',
}


def get_temporal_eval_data(
    benchmarks: List[str],
    data_bucket: str,
    max_samples_per_benchmark: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Load pre-formatted temporal year-prediction eval parquets from GCS.

    Expects parquet files at {data_bucket}/eval/temporal/{benchmark}.parquet
    with columns: encoder_input, candidates, correct_idx, category, benchmark,
    true_year. Produced by prepare_temporal_eval_data.py.

    Args:
        benchmarks: list of benchmark names (e.g. ['temporal_ko', 'temporal_en'])
        data_bucket: GCS bucket prefix (e.g. gs://r4z0kd9han2han)
        max_samples_per_benchmark: max examples per benchmark (deterministic sampling)
        seed: random seed for sampling

    Returns:
        dict mapping benchmark name -> list of normalized eval examples
    """
    import random

    eval_data = {}

    for benchmark in benchmarks:
        if data_bucket is not None:
            parquet_path = f"{data_bucket.rstrip('/')}/eval/temporal/{benchmark}.parquet"
        else:
            parquet_path = f"eval/temporal/{benchmark}.parquet"
        try:
            df = pl.read_parquet(parquet_path)
        except Exception as e:
            logger.error(f"Failed to load temporal eval {benchmark} from {parquet_path}: {e}")
            continue

        examples = df.to_dicts()
        if not examples:
            logger.warning(f"Empty temporal eval data for {benchmark} at {parquet_path}")
            continue

        missing = TEMPORAL_EVAL_REQUIRED_FIELDS - set(examples[0].keys())
        if missing:
            raise ValueError(
                f"Temporal eval parquet '{parquet_path}' missing required fields: {missing}. "
                f"Available: {list(examples[0].keys())}. "
                f"Re-run prepare_temporal_eval_data.py to regenerate."
            )

        if len(examples) > max_samples_per_benchmark:
            rng = random.Random(seed)
            examples = rng.sample(examples, max_samples_per_benchmark)

        eval_data[benchmark] = examples
        logger.info(
            f"Temporal eval: loaded {len(examples)} examples for {benchmark} from {parquet_path}"
        )

    return eval_data


def get_local_sft_datasets(
    base_dir: str = "task_data",
    sft_tasks: str = "all",
    split: str = "train",
    data_bucket: Optional[str] = None,
    host_idx: Optional[int] = None,
    num_hosts: Optional[int] = None,
    sampling_strategy: str = "manual",
    sampling_temperature: float = 2.0,
    sampling_cap: Optional[int] = None,
    sampling_cap_multiplier: Optional[float] = None,
    force_reload: bool = False,
) -> tuple[Dict[str, Dataset], Dict[str, float], Dict[str, DataSourceConfig]]:
    """Load SFT datasets for fine-tuning.

    Supports both local files (task_data/) and GCS paths when data_bucket is set.
    When data_bucket is provided, base_dir is interpreted relative to the bucket
    (e.g., data_bucket='gs://my-bucket' -> 'gs://my-bucket/task_data/...').

    When host_idx and num_hosts are both provided, each dataset is sliced into
    num_hosts non-overlapping shards and only the shard for host_idx is returned.
    Use this for training data so each host in a multi-host setup sees unique
    examples during gradient all-reduce. Do NOT use for validation data.

    Sampling ratios across the loaded sources are derived per `sampling_strategy`:

        - 'manual'      : use the hand-tuned `source.weight` values (legacy).
        - 'temperature' : examples-proportional with temperature smoothing,
                          p_i = N_i^(1/T) / sum_j N_j^(1/T). T=1 is natural,
                          T->inf is uniform. T is `sampling_temperature`.
        - 'capped'      : FLAN-style examples-proportional with a per-task cap K,
                          p_i = min(N_i, K) / sum_j min(N_j, K). Either supply K
                          directly via `sampling_cap`, or auto-compute it as
                          `sampling_cap_multiplier * min_i N_i` (measured BEFORE
                          per-host slicing, so it's stable across host counts).

    Args:
        base_dir: Base directory containing task subdirectories
        sft_tasks: Which SFT tasks to include ('all' or comma-separated list
                   like 'nli,sts,topic_classification,temporal_classification')
        split: Data split to use ('train', 'validation', 'test')
        data_bucket: GCS bucket URI (e.g., 'gs://my-bucket'). When set, reads
                     from GCS instead of local disk.
        host_idx: Index of this host (0-based). Both host_idx and num_hosts
                  must be set to enable per-host slicing.
        num_hosts: Total number of hosts. Both must be set to enable slicing.
        sampling_strategy: 'manual', 'temperature', or 'capped'.
        sampling_temperature: T for temperature mixing. Ignored unless
                              strategy='temperature'.
        sampling_cap: Explicit per-task cap K. Ignored unless strategy='capped'.
        sampling_cap_multiplier: If set with strategy='capped', auto K =
                                 multiplier * min(N_i). Mutually exclusive with
                                 sampling_cap.
        force_reload: If True, bypass the per-host SFT parquet cache and
                      re-slice from source parquets.

    Returns:
        tuple of (datasets dict, sampling_ratios dict, source_configs dict)
    """
    effective_base = f'{data_bucket}/{base_dir}' if data_bucket else base_dir
    sources = create_local_sft_sources(
        base_dir=effective_base, sft_tasks=sft_tasks, split=split
    )

    if not sources:
        log_from_all_processes(logger, 'info', "No local SFT sources configured")
        return {}, {}, {}

    # cache layout: separate sft_data_cache namespace, GCS-backed when a bucket
    # is provided and local (./task_data_cache) otherwise. epath handles both.
    cache_dir = (
        f"{data_bucket.rstrip('/')}/sft_data_cache" if data_bucket else "task_data_cache"
    )
    loader = DynamicDataLoader(
        cache_dir=cache_dir,
        data_bucket=data_bucket or "",
    )

    # treat single-host (or unspecified host_idx) as host 0 of 1 so the cache
    # key shape stays uniform and single-host runs still benefit from caching.
    effective_host_idx = host_idx if host_idx is not None else 0
    effective_num_hosts = num_hosts if num_hosts is not None and num_hosts > 0 else 1

    datasets = {}
    full_sizes: Dict[str, int] = {}
    for source in sources:
        try:
            ds, full_size = loader.load_sft_source_slice(
                source_config=source,
                host_idx=effective_host_idx,
                num_hosts=effective_num_hosts,
                split=split,
                force_reload=force_reload,
            )
            datasets[source.name] = ds
            full_sizes[source.name] = full_size
            log_from_all_processes(logger, 'info',
                f"Loaded local SFT {source.name}: {len(ds):,} examples "
                f"(host {effective_host_idx}/{effective_num_hosts}, "
                f"full_size={full_size:,}, type={source.data_type})"
            )
        except FileNotFoundError as e:
            log_from_all_processes(logger, 'warning', f"Skipping {source.name}: {e}")

    # set custom info attrs (load_sft_source_slice returns a fresh memory-mapped
    # Dataset, so attaching custom attrs is safe -- no select()-copy hazard).
    loaded_source_map = {s.name: s for s in sources if s.name in datasets}
    for name, ds in datasets.items():
        source = loaded_source_map[name]
        ds.info.source_type = source.name
        ds.info.data_type = source.data_type

    # calculate sampling ratios from loaded sources only
    loaded_sources = [s for s in sources if s.name in datasets]
    sampling_ratios = _compute_sft_sampling_ratios(
        loaded_sources=loaded_sources,
        full_sizes=full_sizes,
        strategy=sampling_strategy,
        temperature=sampling_temperature,
        cap=sampling_cap,
        cap_multiplier=sampling_cap_multiplier,
    )

    source_configs = {source.name: source for source in loaded_sources}

    return datasets, sampling_ratios, source_configs


def _compute_sft_sampling_ratios(
    loaded_sources: List[DataSourceConfig],
    full_sizes: Dict[str, int],
    strategy: str,
    temperature: float,
    cap: Optional[int],
    cap_multiplier: Optional[float],
) -> Dict[str, float]:
    """Compute per-source sampling ratios under the requested mixing strategy.

    See get_local_sft_datasets() for strategy semantics. full_sizes must
    reflect dataset sizes BEFORE any per-host slicing so cap/temperature
    math is independent of host count.
    """
    names = [s.name for s in loaded_sources]

    if strategy == "manual":
        total_weight = sum(s.weight for s in loaded_sources)
        if total_weight <= 0:
            raise ValueError(
                f"manual sampling requires positive total weight, got {total_weight}"
            )
        return {s.name: s.weight / total_weight for s in loaded_sources}

    sizes = {name: full_sizes[name] for name in names}
    if any(n <= 0 for n in sizes.values()):
        raise ValueError(f"non-positive dataset size in {sizes}")

    if strategy == "temperature":
        if temperature <= 0:
            raise ValueError(
                f"sampling_temperature must be > 0, got {temperature}"
            )
        powered = {n: (N ** (1.0 / temperature)) for n, N in sizes.items()}
        total = sum(powered.values())
        ratios = {n: v / total for n, v in powered.items()}
        log_from_all_processes(logger, 'info',
            f"SFT sampling: temperature T={temperature} over sizes={sizes}"
        )
        return ratios

    if strategy == "capped":
        if cap is None and cap_multiplier is None:
            raise ValueError(
                "capped sampling requires sampling_cap or sampling_cap_multiplier"
            )
        if cap is not None and cap_multiplier is not None:
            raise ValueError(
                "provide only one of sampling_cap, sampling_cap_multiplier"
            )
        if cap is not None:
            if cap <= 0:
                raise ValueError(f"sampling_cap must be > 0, got {cap}")
            K = int(cap)
            log_from_all_processes(logger, 'info',
                f"SFT sampling: capped K={K:,} (explicit) over sizes={sizes}"
            )
        else:
            if cap_multiplier <= 0:
                raise ValueError(
                    f"sampling_cap_multiplier must be > 0, got {cap_multiplier}"
                )
            min_size = min(sizes.values())
            K = max(1, int(round(min_size * cap_multiplier)))
            log_from_all_processes(logger, 'info',
                f"SFT sampling: capped K={K:,} "
                f"(auto: {cap_multiplier} * min(N)={min_size:,}) over sizes={sizes}"
            )
        effective = {n: min(N, K) for n, N in sizes.items()}
        total = sum(effective.values())
        return {n: v / total for n, v in effective.items()}

    raise ValueError(
        f"Unknown sampling_strategy: {strategy!r} "
        f"(expected 'manual', 'temperature', or 'capped')"
    )