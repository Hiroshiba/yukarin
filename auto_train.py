import argparse
import json
import optuna
from copy import deepcopy
from functools import partial
from optuna.integration import ChainerPruningExtension
from pathlib import Path
from typing import Any, Dict

from yukarin.config import create_from_dict
from yukarin.trainer import create_trainer


def _objective(
        trial: optuna.Trial,
        root_output: Path,
        base_config_dict: Dict[str, Any],
        auto_tune_dataset: bool,
        auto_tune_model: bool,
        auto_tune_optimizer: bool,
):
    config_dict = deepcopy(base_config_dict)

    # dataset
    if auto_tune_dataset:
        config_dict['dataset']['input_global_noise'] = trial.suggest_loguniform('d_ign', 1e-3, 1e+0)
        config_dict['dataset']['input_local_noise'] = trial.suggest_loguniform('d_iln', 1e-3, 1e+0)
        config_dict['dataset']['target_global_noise'] = trial.suggest_loguniform('d_tgn', 1e-3, 1e+0)
        config_dict['dataset']['target_local_noise'] = trial.suggest_loguniform('d_tln', 1e-3, 1e+0)

    # modelo
    if auto_tune_model:
        config_dict['model']['generator_base_channels'] = 2 ** trial.suggest_int('m_gbce', 0, 6)
        config_dict['model']['generator_extensive_layers'] = trial.suggest_int('m_gel', 0, 8)

    # optimizer
    if auto_tune_optimizer:
        name = trial.suggest_categorical('o_n', ['adam', 'sgd'])
        config_dict['train']['optimizer']['name'] = name
        if name == 'adam':
            config_dict['train']['optimizer']['alpha'] = trial.suggest_loguniform('o_a', 1e-4, 3e-3)
        elif name == 'sgd':
            config_dict['train']['optimizer']['lr'] = trial.suggest_loguniform('o_l', 3e-4, 1e-2)
        else:
            raise ValueError(name)

    # train
    postfix = '-'.join(
        f'{k}={v:.2e}' if isinstance(v, float) else f'{k}={v}'
        for k, v in sorted(trial.params.items())
    )

    config = create_from_dict(config_dict)
    output = root_output / (f'{trial.number}-' + postfix)
    output.mkdir(exist_ok=True)

    trainer = create_trainer(config, output)
    trainer.extend(ChainerPruningExtension(trial, 'test/predictor/loss', (config.train.log_iteration, 'iteration')))
    trainer.run()

    log_last = trainer.get_extension('LogReport').log[-1]
    for key, value in log_last.items():
        trial.set_user_attr(key, value)

    return log_last['test/predictor/loss']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_path', type=Path)
    parser.add_argument('root_output', type=Path)
    parser.add_argument('--name', default='yukarin')
    parser.add_argument('--storage', default='sqlite:///optuna.db')
    parser.add_argument('--num_trials', type=int)
    parser.add_argument('--auto_tune_dataset', action='store_true')
    parser.add_argument('--auto_tune_model', action='store_true')
    parser.add_argument('--auto_tune_optimizer', action='store_true')
    arguments = parser.parse_args()

    arguments.root_output.mkdir(exist_ok=True)

    base_config_dict = json.load(open(arguments.config_json_path))
    objective = partial(
        _objective,
        root_output=arguments.root_output,
        base_config_dict=base_config_dict,
        auto_tune_dataset=arguments.auto_tune_dataset,
        auto_tune_model=arguments.auto_tune_model,
        auto_tune_optimizer=arguments.auto_tune_optimizer,
    )

    study = optuna.create_study(study_name=arguments.name, storage=arguments.storage, load_if_exists=True)
    study.optimize(func=objective, n_trials=arguments.num_trials)


if __name__ == '__main__':
    main()
