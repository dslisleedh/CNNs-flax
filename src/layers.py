import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import einops

from typing import Sequence, Tuple, Union, Optional


class LocalResponsibleNormalization(nn.Module):
    k: int = 2
    alpha: float = 1.
    beta: float = .75
    across_channel: bool = True

    @nn.compact
    def __call__(self, x):
        div = jnp.power(x, 2)
        if self.across_channel:
            div = jnp.expand_dims(div, axis=-2)
            div = nn.avg_pool(div, window_shape=(1, 1, self.k), strides=(1, 1, 1), padding='SAME')
            div = div.squeeze()

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
    act = staticmethod(nn.relu)

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


class SkipBlock(nn.Module):
    forward: nn.Module
    act = staticmethod(nn.relu)

    @nn.compact
    def __call__(self, x, training: bool):
        if self.forward.down_sample:
            skip = nn.Conv(self.forward.n_filters[-1], (3, 3), strides=(2, 2), padding='SAME')(x)
        else:
            if self.forward.increase_dim:
                skip = nn.Conv(self.forward.n_filters[-1], (1, 1), padding='SAME')(x)
            else:
                skip = x

        x = self.forward(x, training=training)
        x = self.act(x + skip)
        return x


class ResBlock(nn.Module):
    n_filters: Sequence[int]
    act = staticmethod(nn.relu)
    down_sample: bool = False
    increase_dim: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(
            self.n_filters[0], (3, 3) if len(self.n_filters) == 2 else (1, 1), padding='SAME',
            strides=(2, 2) if self.down_sample else (1, 1), use_bias=False
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        for i, n_filter in enumerate(self.n_filters[1:]):
            x = self.act(x)
            x = nn.Conv(n_filter, (3, 3) if i == 0 else (1, 1), padding="SAME", use_bias=False)(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
        return x


class IdentitySkipBlock(nn.Module):
    forward: nn.Module
    act = staticmethod(nn.relu)

    @nn.compact
    def __call__(self, x, training: bool):
        if self.forward.increase_dim:
            x = nn.BatchNorm()(x, use_running_average=not training)
            x = self.act(x)
            skip = nn.Conv(self.forward.n_filters[-1], (1, 1), padding='SAME')(x)
        else:
            if self.forward.down_sample:
                skip = nn.Conv(self.forward.n_filters[-1], (3, 3), strides=(2, 2), padding='SAME')(x)
            else:
                skip = x
        x = self.forward(x, training) + skip
        return x


class PreActResBlock(nn.Module):
    n_filters: Sequence[int]
    act = staticmethod(nn.relu)
    down_sample: bool = False
    increase_dim: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        if not self.increase_dim:
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = self.act(x)
        x = nn.Conv(
            self.n_filters[0], (1, 1), padding='SAME', use_bias=False,
            strides=(2, 2) if self.down_sample else (1, 1)
        )(x)
        for i, n_filter in enumerate(self.n_filters[1:]):
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = self.act(x)
            x = nn.Conv(n_filter, (3, 3) if i == 0 else (1, 1), padding="SAME", use_bias=False)(x)
        return x


class DenseLayer(nn.Module):
    n_filters: int
    act = staticmethod(nn.relu)
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
        return y


class DenseTransitionLayer(nn.Module):
    n_filters: int
    act = staticmethod(nn.relu)
    dropout_rate: float = .2

    @nn.compact
    def __call__(self, x, training: bool):
        y = nn.BatchNorm()(x, use_running_average=not training)
        y = self.act(y)
        y = nn.Conv(self.n_filters, (1, 1), padding="VALID")(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not training)
        y = nn.avg_pool(y, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return y


class SEBlock(nn.Module):
    forward: nn.Module
    r: int = 16

    @nn.compact
    def __call__(self, x, training: bool):
        y = self.forward(x, training=training)

        n_filters = y.shape[-1]
        attention = jnp.mean(y, axis=(1, 2), keepdims=True)
        attention = nn.Dense(n_filters // self.r)(attention)
        attention = nn.relu(attention)
        attention = nn.Dense(n_filters)(attention)
        attention = nn.sigmoid(attention)

        return y * attention


class AggResBlock(nn.Module):
    n_filters: Sequence[int]
    act = staticmethod(nn.relu)
    down_sample: bool = False
    increase_dim: bool = False
    c: int = 32
    '''
    TODO: Maybe LocalConv3D is better than list of Conv2D ?
    '''

    @nn.compact
    def __call__(self, x, training: bool):
        x = [
            nn.Conv(
                self.n_filters[0] // self.c, (3, 3) if len(self.n_filters) == 2 else (1, 1), padding='SAME',
                strides=(2, 2) if self.down_sample else (1, 1), use_bias=False
            )(x) for _ in range(self.c)
        ]
        x = [nn.BatchNorm(use_running_average=not training)(x_) for x_ in x]
        for i, n_filter in enumerate(self.n_filters[1:]):
            # List comprehension is faster than stack/activation/split
            x = [self.act(x_) for x_ in x]
            x = [
                nn.Conv(
                    n_filter // self.c if i == 0 else n_filter, (3, 3) if i == 0 else (1, 1), padding="SAME",
                    use_bias=False
                )(x_) for x_ in x
            ]
            x = [nn.BatchNorm()(x_, use_running_average=not training) for x_ in x]

        x = jnp.stack(x, axis=-1).sum(axis=-1)
        return x


class SESkipBlock(nn.Module):
    forward: nn.Module
    act = staticmethod(nn.relu)
    r: int = 16

    @nn.compact
    def __call__(self, x, training: bool):
        if self.forward.increase_dim:
            x = nn.BatchNorm()(x, use_running_average=not training)
            x = self.act(x)
            skip = nn.Conv(self.forward.n_filters[-1], (1, 1), padding='VALID')(x)
        else:
            if self.forward.down_sample:
                skip = nn.Conv(self.forward.n_filters[-1], (3, 3), strides=(2, 2), padding='SAME')(x)
            else:
                skip = x

        y = self.forward(x, training=training)

        n_filters = y.shape[-1]
        attention = jnp.mean(x, axis=(1, 2), keepdims=True)
        attention = nn.Dense(n_filters // self.r)(attention)
        attention = nn.relu(attention)
        attention = nn.Dense(n_filters)(attention)
        attention = nn.sigmoid(attention)

        return y * attention + skip


class DynamicConv2D(nn.Module):
    n_filters: int
    kernel_size: Union[int, Sequence[int]] = (3, 3)
    strides: Union[int, Sequence[int]] = (1, 1)
    padding: str = 'SAME'
    feature_group_count: int = 1
    use_bias: bool = True
    act = staticmethod(nn.relu)

    r_filters: int = 4
    n_kernels: int = 4
    initial_temperature: float = 30.
    temperature_decay: float = 1e-4

    @nn.compact
    def __call__(self, x, training: bool):
        input_batches, _, _, input_filters = x.shape

        if type(self.kernel_size) == int:
            kernel_size = (self.kernel_size,) * 2
        else:
            kernel_size = self.kernel_size

        if type(self.strides) == int:
            strides = (self.strides,) * 2
        else:
            strides = self.strides

        reduction_kernel = self.param(
            'reduction_kernel', nn.initializers.variance_scaling(.02, mode='fan_in', distribution='truncated_normal'),
            (input_filters, self.r_filters)
        )
        attention_kernel = self.param(
            'attention_kernel', nn.initializers.variance_scaling(.02, mode='fan_in', distribution='truncated_normal'),
            (self.r_filters, self.n_kernels)
        )
        kernels = self.param(
            'conv_kernels', nn.initializers.variance_scaling(.02, mode='fan_in', distribution='truncated_normal'),
            (1, self.n_kernels,) + kernel_size + (input_filters, self.n_filters,)
        )
        temperature = self.variable(
            'state', 'temperature', lambda s: jnp.ones(s, jnp.float32) * 30, (1,)
        )

        pool = jnp.mean(x, axis=(1, 2))
        pool_reduction = self.act(
            lax.dot_general(pool, reduction_kernel, (((1,), (0,)), ((), ())))
        )
        attention = nn.softmax(
            lax.dot_general(pool_reduction, attention_kernel, (((1,), (0,)), ((), ()))) / temperature.value
        )
        attention = attention[:, :, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        w_tiled = jnp.sum(
            attention * kernels, axis=1
        )
        w_tiled = einops.rearrange(
            w_tiled, 'B H W F_in F_out -> H W F_in (B F_out)'
        )
        x = jnp.expand_dims(
            einops.rearrange(
                x, 'B H W C -> H W (B C)'
            ), axis=0
        )
        y = lax.conv_general_dilated(
            x, w_tiled, window_strides=strides, padding=self.padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=input_batches
        )
        y = einops.rearrange(
            y[0], 'H W (B C) -> B H W C', B=input_batches
        )

        if self.use_bias:
            bias = self.param(
                'bias', nn.initializers.zeros, (1, self.n_kernels, self.n_filters)
            )
            attention = jnp.squeeze(
                attention, axis=[3, 4, 5]
            )
            b_tiled = jnp.sum(attention * bias, axis=1)
            b_tiled = jnp.reshape(
                b_tiled, (input_batches, 1, 1, self.n_filters)
            )
            y = y + b_tiled

        y = nn.BatchNorm()(y, use_running_average=not training)
        y = self.act(y)

        if training:
            temperature.value = max(
                1, temperature.value - self.temperature_decay
            )

        return y