import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn

from typing import Sequence, Tuple, Union, Optional


class LocalResponsibleNormalization(nn.Module):
    k: int = 2
    alpha: float = 1.
    beta: float = .75
    across_channel: bool = True

    @nn.compact
    def __call__(self, x):
        div = x.pow(2)
        if self.across_channel:
            div = div.expand_dims(1)
            div = nn.avg_pool(div, window_shape=(self.k, 1, 1), strides=(1, 1, 1), padding=(int(self.k / 2), 0, 0))
            div = div.squeeze(1)

        else:
            div = nn.avg_pool(div, window_shape=(self.k, self.k), strides=(1, 1), padding="SAME")
        div = jnp.power(div * self.alpha + 1., self.beta)

        return x / div


class Inception(nn.Module):
    conv_1_filters: int
    conv_3_filters: Sequence[int]
    conv_5_filters: Sequence[int]
    max_pool_filters: int
    down_sample: bool = False
    act = nn.relu

    @nn.compact
    def __call__(self, x):
        if self.downsamle:
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        conv_1 = nn.Conv(self.conv_1_filters, (1, 1), padding="SAME")(x)
        conv_1 = self.act(conv_1)

        conv_3 = nn.Conv(self.conv_3_filters[0], (1, 1), padding="SAME")(x)
        conv_3 = self.act(conv_3)
        conv_3 = nn.Conv(self.conv_3_filters[1], (3, 3), padding="SAME")(conv_3)
        conv_3 = self.act(conv_3)

        conv_5 = nn.Conv(self.conv_5_filters[0], (1, 1), padding="SAME")(x)
        conv_5 = self.act(conv_5)
        conv_5 = nn.Conv(self.conv_5_filters[1], (5, 5), padding="SAME")(conv_5)
        conv_5 = self.act(conv_5)

        max_pool = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")
        max_pool = nn.Conv(self.max_pool_filters, (1, 1), padding="SAME")(max_pool)
        max_pool = self.act(max_pool)

        return jnp.concatenate([conv_1, conv_3, conv_5, max_pool], axis=-1)


class ResBlock(nn.Module):
    n_filters: Sequence[int]
    act = nn.relu
    down_sample: bool = False
    increase_dim: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        if self.down_sample:
            skip = nn.Conv(self.n_filters[-1], (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.Conv(
                self.n_filters[0], (3, 3) if len(self.n_filters) == 2 else (1, 1), strides=(2, 2),
                padding="SAME", use_bias=False
            )(x)
        else:
            if self.increase_dim:
                skip = nn.Conv(self.n_filters[-1], (1, 1), padding="SAME")(x)
            else:
                skip = jnp.identity(x)
            x = nn.Conv(
                self.n_filters[0], (3, 3) if len(self.n_filters) == 2 else (1, 1),
                padding="SAME", use_bias=False
            )(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        for i, n_filter in enumerate(self.n_filters[1:]):
            x = self.act(x)
            x = nn.Conv(n_filter, (3, 3) if i == 0 else (1, 1), padding="SAME", use_bias=False)(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
        x += skip
        x = self.act(x)
        return x


class IdentityResBlock(nn.Module):
    n_filters: Sequence[int]
    act = nn.relu
    down_sample: bool = False
    increase_dim: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        if self.down_sample:
            skip = nn.Conv(self.n_filters[-1], (1, 1), strides=(2, 2), padding="SAME")(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
            x = self.act(x)
            x = nn.Conv(
                self.n_filters[0], (3, 3) if len(self.n_filters == 2) else (1, 1), strides=(2, 2),
                padding="SAME", use_bias=False
            )(x)
        else:
            if not self.increase_dim:
                skip = jnp.identity(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
            x = self.act(x)
            if self.increase_dim:
                skip = nn.Conv(self.n_filters[-1], (1, 1), padding="SAME")(x)
            x = nn.Conv(
                self.n_filters[0], (3, 3) if len(self.n_filters == 2) else (1, 1), padding="SAME"
            )(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        for i, n_filter in enumerate(self.n_filters[1:]):
            x = self.act(x)
            x = nn.Conv(self.n_filters, (3, 3) if i == 0 else (1, 1), padding="SAME")(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
        x = x + skip
        return x


class DenseLayer(nn.Module):
    n_filters: int
    act = nn.relu
    dropout_rate: float = .2

    @nn.compact
    def __call__(self, x, training: bool):
        y = nn.BatchNorm()(x, use_running_average=not training)
        y = self.act(y)
        y = nn.Conv(self.n_filters * 4, (1, 1), padding="VALID")(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not training)
        y = nn.BatchNorm()(y, use_running_average=not training)
        y = self.act(y)
        y = nn.Conv(self.n_filters, (3, 3), padding="SAME")(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not training)
        return jnp.concatenate([x, y], axis=-1)


class DenseTransitionLayer(nn.Module):
    n_filters: int
    act = nn.relu
    dropout_rate: float = .2

    @nn.compact
    def __call__(self, x, training: bool):
        y = nn.BatchNorm()(x, use_running_average=not training)
        y = self.act(y)
        y = nn.Conv(self.n_filters, (1, 1), padding="VALID")(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not training)
        y = nn.avg_pool(y, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return y

