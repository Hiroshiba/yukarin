from typing import List

import chainer
import chainer.functions as F
import numpy

from yukarin.config import LossConfig
from yukarin.model import Discriminator
from yukarin.model import Predictor


def _loss_predictor(predictor, output, target, d_fake, loss_config: LossConfig):
    b, _, t = d_fake.data.shape

    loss_mse = (F.mean_absolute_error(output, target))
    chainer.report({'mse': loss_mse}, predictor)

    loss_adv = F.sum(F.softplus(-d_fake)) / (b * t)
    chainer.report({'adversarial': loss_adv}, predictor)

    loss = loss_config.mse * loss_mse + loss_config.adversarial * loss_adv
    chainer.report({'loss': loss}, predictor)
    return loss


def _loss_discriminator(discriminator: Discriminator, d_real: chainer.Variable, d_fake: chainer.Variable):
    b, _, t = d_real.data.shape

    loss_real = F.sum(F.softplus(-d_real)) / (b * t)
    chainer.report({'real': loss_real}, discriminator)

    loss_fake = F.sum(F.softplus(d_fake)) / (b * t)
    chainer.report({'fake': loss_fake}, discriminator)

    loss = loss_real + loss_fake
    chainer.report({'loss': loss}, discriminator)

    tp = (d_real.data > 0.5).sum()
    fp = (d_fake.data > 0.5).sum()
    fn = (d_real.data <= 0.5).sum()
    tn = (d_fake.data <= 0.5).sum()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    chainer.report({'accuracy': accuracy}, discriminator)
    chainer.report({'precision': precision}, discriminator)
    chainer.report({'recall': recall}, discriminator)
    return loss


class Updater(chainer.training.StandardUpdater):
    def __init__(
            self,
            loss_config: LossConfig,
            predictor: Predictor,
            discriminator: Discriminator,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.predictor = predictor
        self.discriminator = discriminator

    def forward(self, input, target, mask):
        input = chainer.as_variable(input)
        target = chainer.as_variable(target)
        mask = chainer.as_variable(mask)

        output = self.predictor(input)
        output = output * mask
        target = target * mask

        d_fake = self.discriminator(input, output)
        d_real = self.discriminator(input, target)

        loss = {
            'predictor': _loss_predictor(self.predictor, output, target, d_fake, self.loss_config),
            'discriminator': _loss_discriminator(self.discriminator, d_real, d_fake),
        }
        return loss

    def update_core(self):
        opt_predictor = self.get_optimizer('predictor')
        opt_discriminator = self.get_optimizer('discriminator')

        batch = self.get_iterator('main').next()
        batch = self.converter(batch, self.device)
        loss = self.forward(**batch)

        opt_predictor.update(loss.get, 'predictor')
        opt_discriminator.update(loss.get, 'discriminator')


def _loss_predictor_cg(predictor, reconstruct, output, target, d_fake, loss_config: LossConfig):
    b, _, t = d_fake.data.shape

    loss_mse = (F.mean_absolute_error(reconstruct, target))
    chainer.report({'mse': loss_mse}, predictor)

    loss_identity = (F.mean_absolute_error(output, target))
    chainer.report({'identity': loss_identity}, predictor)

    loss_adv = F.sum(F.softplus(-d_fake)) / (b * t)
    chainer.report({'adversarial': loss_adv}, predictor)

    loss = loss_config.mse * loss_mse + loss_config.mse / 100 * loss_identity + loss_config.adversarial * loss_adv
    chainer.report({'loss': loss}, predictor)
    return loss


class CGUpdater(chainer.training.StandardUpdater):
    def __init__(
            self,
            loss_config: LossConfig,
            predictor_xy: Predictor,
            predictor_yx: Predictor,
            discriminator_x: Discriminator,
            discriminator_y: Discriminator,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.predictor_xy = predictor_xy
        self.predictor_yx = predictor_yx
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        self._max_buffer_size = 50
        self._buffer_x: List[numpy.ndarray] = []
        self._buffer_y: List[numpy.ndarray] = []

    def _get_and_update_buffer(self, data, buffer: List[numpy.ndarray]):
        buffer.append(data)

        if len(buffer) <= self._max_buffer_size:
            return data

        buffer.pop(0)

        if numpy.random.rand() < 0.5:
            return data

        return buffer[numpy.random.randint(0, self._max_buffer_size)]

    def forward(self, x, y, mask_x, mask_y):
        x = chainer.as_variable(x)
        y = chainer.as_variable(y)
        mask_x = chainer.as_variable(mask_x)
        mask_y = chainer.as_variable(mask_y)

        x_y = self.predictor_xy(x) * mask_x
        x_y_x = self.predictor_yx(x_y) * mask_x
        x_y_buffer = chainer.as_variable(self._get_and_update_buffer(x_y.data, self._buffer_x))

        y_x = self.predictor_yx(y) * mask_y
        y_x_y = self.predictor_xy(y_x) * mask_y
        y_x_buffer = chainer.as_variable(self._get_and_update_buffer(y_x.data, self._buffer_y))

        dx_real = self.discriminator_x(x)
        dx_fake = self.discriminator_x(y_x)
        dx_fake_buffer = self.discriminator_x(y_x_buffer)

        dy_real = self.discriminator_y(y)
        dy_fake = self.discriminator_y(x_y)
        dy_fake_buffer = self.discriminator_y(x_y_buffer)

        l_p_x = _loss_predictor_cg(self.predictor_yx, x_y_x, x_y, x, dx_fake, self.loss_config)
        l_p_y = _loss_predictor_cg(self.predictor_xy, y_x_y, y_x, y, dy_fake, self.loss_config)
        loss_predictor = l_p_x + l_p_y

        loss = {
            'predictor': loss_predictor,
            'discriminator_x': _loss_discriminator(self.discriminator_x, dx_real, dx_fake_buffer),
            'discriminator_y': _loss_discriminator(self.discriminator_y, dy_real, dy_fake_buffer),
        }
        return loss

    def update_core(self):
        batch = self.get_iterator('main').next()
        batch = self.converter(batch, self.device)
        loss = self.forward(**batch)

        self.get_optimizer('predictor_xy').update(loss.get, 'predictor')
        self.get_optimizer('predictor_yx').update(loss.get, 'predictor')
        self.get_optimizer('discriminator_x').update(loss.get, 'discriminator_x')
        self.get_optimizer('discriminator_y').update(loss.get, 'discriminator_y')
