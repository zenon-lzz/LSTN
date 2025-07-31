"""
=================================================
@Author: Zenon
@Date: 2025-04-14
@Description: Main entry point for running time series anomaly detection experiments.
==================================================
"""
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from configs.constants import PROJECT_ROOT, TuningTypeEnum
from experiments.benchmarks.dcdetector_exp import DCdetectorExperiment
from experiments.benchmarks.memto_exp import MemtoExperiment
from experiments.benchmarks.mtscid_exp import MtsCIDExperiment
from experiments.benchmarks.st_fusion_exp import STFusionExperiment
from experiments.benchmarks.times_net_exp import TimesNetExperiment
from experiments.tuning.exp_module_tuning import STFusionModuleTuningExperiment
from tsadlib import constants, log
from tsadlib.configs.log_config import configure_global_logger
from tsadlib.utils.files import write_to_csv
from tsadlib.utils.format_string import format_args
from tsadlib.utils.gpu import empty_gpu_cache
from tsadlib.utils.parsers import parse_args, parse_basic_config
from tsadlib.utils.randoms import set_random_seed

# Dictionary mapping model names to their corresponding experiment classes
BenchmarksExperiment = {
    'MtsCID': MtsCIDExperiment,
    'MEMTO': MemtoExperiment,
    'TimesNet': TimesNetExperiment,
    'STFusion': STFusionExperiment,
    'DCdetector': DCdetectorExperiment
}

TuningExperiment = {
    'STFusionModuleTuning': STFusionModuleTuningExperiment
}

if __name__ == '__main__':

    # Read logging configuration from INI file
    log_config = parse_basic_config()['LOGGING']

    # Setup logging directory and configure global logger
    log_dir = Path.joinpath(PROJECT_ROOT, log_config['log_dir'])
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    configure_global_logger(log_dir, log_config['level'])

    
    # Set fixed random seed for reproducible results
    set_random_seed(constants.FIX_SEED)

    # Parse command line arguments and format them for logging
    args = parse_args()
    args_string = format_args(args)
    log.success(args_string)

    # Initialize metrics collection and get model name
    metrics = []
    model = args.model

    # tuning type parameters, ignore it if args.task_name is benchmarks
    # tuning_type = TuningTypeEnum.TEMPORAL_MEMORY
    tuning_type = TuningTypeEnum(args.tuning_type)

    # Run experiments for specified number of iterations
    for run in range(args.runs):
        # Create unique setting identifier for this run
        setting = f'{args.task_name}_{model}_{args.dataset}_iter{run + 1}'
        if args.task_name == 'tuning':
            exp = TuningExperiment[model](args, tuning_type)
        else:
            # benchmarks experiment
            exp = BenchmarksExperiment[model](args)

        if args.mode == 'train':
            # Training mode: train the model first, then test
            log.info(
                f'\n>>>>>>>{run + 1:>3}/{args.runs:>3} {args.model} starts training in {args.dataset} dataset: >>>>>>>>>>>>>>>>>>>>>>>>>>')
            start_time = time.time()
            exp.train(setting)
            log.info(f'Training costs time: {time.time() - start_time:10.2}s.')

            # Test the trained model
            log.info(f'\n>>>>>>>{run + 1:>3}/{args.runs:>3} testing: >>>>>>>>>>>>>>>>>>>>>>>>>>')
            start_time = time.time()
            result = exp.test(setting)
            metrics.append(result)
            log.info(f'Testing costs time: {time.time() - start_time:10.2}s.')
        else:
            # Test mode: only test with pre-trained model
            log.info(f'\n>>>>>>>{run + 1:>3}/{args.runs:>3} testing: >>>>>>>>>>>>>>>>>>>>>>>>>>')
            start_time = time.time()
            result = exp.test(setting)
            metrics.append(result)
            log.info(f'Testing costs time: {time.time() - start_time:10.2}s.')

        # Clear GPU cache to prevent memory issues
        empty_gpu_cache()

    # Convert metrics to DataFrame for analysis
    df = pd.DataFrame(metrics)
    log.info(f'\n\n======================={model} Evaluation Results in {args.dataset} Dataset=======================')

    # Setup result file path and write header information
    result_path = os.path.join('results', model, tuning_type.value if args.task_name == 'tuning' else '', f'{args.dataset}.csv')
    write_to_csv(result_path,
                 f'\n\n======================={model} Evaluation Results in {args.dataset} Dataset at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}=======================\n'
                 f'{args_string}'
                 f'============================================================================================')

    # Log and save all individual run results
    log.success('All running result:\n{:s}', df.to_string())
    write_to_csv(result_path, df.round(4).to_string())

    # Calculate statistical summary (mean and standard deviation)
    mean = df.mean().round(4)
    # Check if there are multiple runs to calculate std, otherwise set to 0
    std_dev = df.std().round(4) if len(df) > 1 else pd.Series(0, index=df.columns)

    # Create average result with mean ± std format
    average_result = pd.DataFrame(columns=df.columns)
    row_name = 'average result (Mean ± Std)'
    for col in average_result.columns:
        average_result.loc[row_name, col] = f"{mean[col]} ± {std_dev[col]}"

    # Log and save average results with statistical summary
    log.success('Average running result:\n{:s}', average_result.to_string())
    write_to_csv(result_path, f'Average running result:\n{average_result.to_string()}\n')
