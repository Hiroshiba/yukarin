import argparse
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict

from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.dataset import convert
from chainer.iterators import MultiprocessIterator
from chainer.training import extensions
from tb_chainer import SummaryWriter

from utility.chainer_utility import TensorBoardReport
from yukarin.config import create_from_json
from yukarin.dataset import create as create_dataset
from yukarin.model import create
from yukarin.updater import Updater

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_path', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()

    config = create_from_json(arguments.config_json_path)
    arguments.output.mkdir(exist_ok=True)
    config.save_as_json((arguments.output / 'config.json').absolute())

    # model
    if config.train.gpu >= 0:
        cuda.get_device_from_id(config.train.gpu).use()
    predictor, discriminator = create(config.model)
    models = {
        'predictor': predictor,
        'discriminator': discriminator,
    }

    if config.train.pretrained_model is not None:
        serializers.load_npz(str(config.train.pretrained_model), predictor)

    # dataset
    dataset = create_dataset(config.dataset)
    train_iter = MultiprocessIterator(dataset['train'], config.train.batchsize)
    test_iter = MultiprocessIterator(dataset['test'], config.train.batchsize, repeat=False, shuffle=False)
    train_eval_iter = MultiprocessIterator(dataset['train_eval'], config.train.batchsize, repeat=False, shuffle=False)


    # optimizer
    def create_optimizer(model):
        cp: Dict[str, Any] = copy(config.train.optimizer)
        n = cp.pop('name').lower()

        if n == 'adam':
            optimizer = optimizers.Adam(**cp)
        elif n == 'sgd':
            optimizer = optimizers.SGD(**cp)
        else:
            raise ValueError(n)

        optimizer.setup(model)
        return optimizer


    opts = {key: create_optimizer(model) for key, model in models.items()}

    # updater
    converter = partial(convert.concat_examples, padding=0)
    updater = Updater(
        loss_config=config.loss,
        predictor=predictor,
        discriminator=discriminator,
        device=config.train.gpu,
        iterator=train_iter,
        optimizer=opts,
        converter=converter,
    )

    # trainer
    trigger_log = (config.train.log_iteration, 'iteration')
    trigger_snapshot = (config.train.snapshot_iteration, 'iteration')
    trigger_stop = (config.train.stop_iteration, 'iteration') if config.train.stop_iteration is not None else None

    trainer = training.Trainer(updater, stop_trigger=trigger_stop, out=arguments.output)
    tb_writer = SummaryWriter(Path(arguments.output))

    ext = extensions.Evaluator(test_iter, models, converter, device=config.train.gpu, eval_func=updater.forward)
    trainer.extend(ext, name='test', trigger=trigger_log)
    ext = extensions.Evaluator(train_eval_iter, models, converter, device=config.train.gpu, eval_func=updater.forward)
    trainer.extend(ext, name='train', trigger=trigger_log)

    trainer.extend(extensions.dump_graph('predictor/loss'))

    ext = extensions.snapshot_object(predictor, filename='predictor_{.updater.iteration}.npz')
    trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(TensorBoardReport(writer=tb_writer), trigger=trigger_log)

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    trainer.run()
