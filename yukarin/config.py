import json
from pathlib import Path
from typing import Dict, Any
from typing import List
from typing import NamedTuple
from typing import Union

from yukarin.param import AcousticParam
from yukarin.utility.json_utility import JSONEncoder


class DatasetConfig(NamedTuple):
    acoustic_param: AcousticParam
    input_glob: Path
    target_glob: Path
    indexes_glob: Path
    in_features: List[str]
    out_features: List[str]
    train_crop_size: int
    input_global_noise: float
    input_local_noise: float
    target_global_noise: float
    target_local_noise: float
    seed: int
    num_test: int


class ModelConfig(NamedTuple):
    in_channels: int
    out_channels: int
    generator_base_channels: int
    generator_extensive_layers: int
    discriminator_base_channels: int
    discriminator_extensive_layers: int
    weak_discriminator: bool


class LossConfig(NamedTuple):
    mse: float
    adversarial: float


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    pretrained_model: Path


class ProjectConfig(NamedTuple):
    name: str
    tags: List[str]


class Config(NamedTuple):
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    project: ProjectConfig

    def save_as_json(self, path):
        d = _namedtuple_to_dict(self)
        json.dump(d, open(path, 'w'), indent=2, sort_keys=True, cls=JSONEncoder)


def _namedtuple_to_dict(o: NamedTuple):
    return {
        k: v if not hasattr(v, '_asdict') else _namedtuple_to_dict(v)
        for k, v in o._asdict().items()
    }


def create_from_json(s: Union[str, Path]):
    d = json.load(open(s))
    backward_compatible(d)

    return Config(
        dataset=DatasetConfig(
            acoustic_param=AcousticParam(**d['dataset']['acoustic_param']),
            input_glob=Path(d['dataset']['input_glob']),
            target_glob=Path(d['dataset']['target_glob']),
            indexes_glob=Path(d['dataset']['indexes_glob']),
            in_features=d['dataset']['in_features'],
            out_features=d['dataset']['out_features'],
            train_crop_size=d['dataset']['train_crop_size'],
            input_global_noise=d['dataset']['input_global_noise'],
            input_local_noise=d['dataset']['input_local_noise'],
            target_global_noise=d['dataset']['target_global_noise'],
            target_local_noise=d['dataset']['target_local_noise'],
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        model=ModelConfig(
            in_channels=d['model']['in_channels'],
            out_channels=d['model']['out_channels'],
            generator_base_channels=d['model']['generator_base_channels'],
            generator_extensive_layers=d['model']['generator_extensive_layers'],
            discriminator_base_channels=d['model']['discriminator_base_channels'],
            discriminator_extensive_layers=d['model']['discriminator_extensive_layers'],
            weak_discriminator=d['model']['weak_discriminator'],
        ),
        loss=LossConfig(
            mse=d['loss']['mse'],
            adversarial=d['loss']['adversarial'],
        ),
        train=TrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
            stop_iteration=d['train']['stop_iteration'],
            optimizer=d['train']['optimizer'],
            pretrained_model=d['train']['pretrained_model'],
        ),
        project=ProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )


def backward_compatible(d: Dict):
    if 'features' in d['dataset']:
        d['dataset']['in_features'] = d['dataset']['features']
        d['dataset']['out_features'] = d['dataset']['features']

    if 'optimizer' not in d['train']:
        d['train']['optimizer'] = dict(
            name='Adam',
            alpha=0.0002,
            beta1=0.5,
            beta2=0.999,
        )

    if 'pretrained_model' not in d['train']:
        d['train']['pretrained_model'] = None
