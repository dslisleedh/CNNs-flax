import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import einops

from src.layers import *

from typing import Sequence, Tuple, Union


class GoogLeNet(nn.Module):

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = LocalResponsibleNormalization()(x)
        x = nn.Conv(64, (1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(192, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = LocalResponsibleNormalization()(x)

        x = InceptionBlock(64, (94, 128), (16, 32), 32, downsample=True)(x)      # inception_3a
        x = InceptionBlock(128, (128, 192), (32, 96), 64)(x)                     # inception_3b

        o_1 = InceptionBlock(192, (96, 208), (16, 48), 64, downsample=True)(x)   # inception_4a
        x = InceptionBlock(160, (112, 224), (24, 64), 64)(o_1)                   # inception_4b
        x = InceptionBlock(128, (128, 256), (24, 64), 64)(x)                     # inception_4c
        o_2 = InceptionBlock(112, (144, 288), (32, 64), 64)(x)                   # inception_4d
        x = InceptionBlock(256, (160, 320), (32, 128), 128)(o_2)                 # inception_5e

        x = InceptionBlock(256, (160, 320), (32, 128), 128)(x, downsample=True)  # inception_5a
        x = InceptionBlock(348, (192, 348), (48, 128), 128)(x)                   # inception_5b

        # Classification head
        x = nn.avg_pool(x, window_shape=(7, 7), strides=(1, 1), padding="VALID")
        x = x.flatten()
        x = nn.Dropout(rate=0.4)(x, deterministic=not training)
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)

        if training:
            # Auxiliary classifier 1
            o_1 = nn.avg_pool(o_1, window_shape=(5, 5), strides=(3, 3), padding="VALID")
            o_1 = o_1.flatten()
            o_1 = nn.Dense(1024)(o_1)
            o_1 = nn.relu(o_1)
            o_1 = nn.Dropout(rate=0.7)(o_1, deterministic=False)
            o_1 = nn.Dense(1000)(o_1)
            o_1 = nn.softmax(o_1)

            # Auxiliary classifier 2
            o_2 = nn.avg_pool(o_2, window_shape=(5, 5), strides=(3, 3), padding="VALID")
            o_2 = o_2.flatten()
            o_2 = nn.Dense(1024)(o_2)
            o_2 = nn.relu(o_2)
            o_2 = nn.Dropout(rate=0.7)(o_2, deterministic=False)
            o_2 = nn.Dense(1000)(o_2)
            o_2 = nn.softmax(o_2)

            return y, o_1, o_2

        else:
            return y


class VGGNet(nn.Module):
    confing: dict

    @nn.compact
    def __call__(self, x, training: bool):
        n_filters = self.config["n_filters"]
        # Feature extraction
        for i, n_layers in enumerate(self.config['n_layers']):
            for _ in range(n_layers):
                x = nn.Conv(n_filters, (3, 3), padding="SAME")(x)
                x = nn.relu(x)
            if i == 0 and self.config["use_lrn"]:
                x = LocalResponsibleNormalization()(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
            n_filters *= 2

        # Classification head
        x = einops.rearrange(x, "b h w c -> b (h w) c")
        x = nn.Dropout(rate=0.5)(x, deterministic=not training)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not training)
        x = nn.Dense(4096)(x)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)
        return y


class ResNet(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, training: bool):
        # Intro
        x = nn.Conv(self.config['intro']['n_filters'], (7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        # Feature extraction
        for i, n_layers in enumerate(self.config['n_layers']):
            if i == 0:
                x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
                x = SkipBlock(
                    ResBlock(self.config['n_filters'][i], increase_dim=True)
                )(x, training=training)
            else:
                x = SkipBlock(
                    ResBlock(self.config['n_filters'][i], downsample=True)
                )(x, training=training)
            for _ in range(n_layers - 1):
                x = SkipBlock(
                    ResBlock(self.config['n_filters'][i])
                )(x, training=training)

        # Classification head
        x = nn.avg_pool(x, window_shape=(7, 7), strides=(1, 1), padding="VALID")
        x = x.squeeze(axis=(1, 2))
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)
        return y


class PreActResNet(nn.Module):
    config: dict
    '''
    for later use
    nStages = {16, 64, 128, 256}
    '''

    @nn.compact
    def __call__(self, x, training: bool):
        # Intro
        x = nn.Conv(self.config['intro']['n_filters'], (7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        # Feature extraction
        for i, n_layers in enumerate(self.config['n_layers']):
            if i == 0:
                x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
                x = IdentitySkipBlock(
                    PreActResBlock(self.config['n_filters'][i], increase_dim=True)
                )(x, training=training)
            else:
                x = IdentitySkipBlock(
                    PreActResBlock(self.config['n_filters'][i], downsample=True)
                )(x, training=training)
            for _ in range(n_layers - 1):
                x = IdentitySkipBlock(
                    PreActResBlock(self.config['n_filters'][i])
                )(x, training=training)

        # Classification head
        x = nn.avg_pool(x, window_shape=(7, 7), strides=(1, 1), padding="VALID")
        x = x.squeeze(axis=(1, 2))
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)
        return y


class DenseNet(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, training: bool):
        n_filters = self.config['initial_filters']
        growth_rate = self.config['growth_rate']
        # Intro
        x = nn.Conv(n_filters, (7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Feature extraction
        for i, n_layers in enumerate(self.config['n_layers']):
            x_list = [x]
            for _ in range(n_layers):
                n_filters += growth_rate
                x = DenseBlock(n_filters)(jnp.concatenate(x_list, axis=-1), training=training)
                x_list.append(x)
            n_filters = int(n_filters * self.config['theta'])
            x = jnp.concatenate(x_list, axis=-1)
            if i != len(self.config['n_layers']) - 1:
                x = DenseTransitionBlock(n_filters)(x, training=training)

        # Classification head
        x = nn.avg_pool(x, window_shape=(7, 7), strides=(1, 1), padding="VALID")
        x = x.squeeze(axis=(1, 2))
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)
        return y


class SEResNet(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, training: bool):
        # Intro
        x = nn.Conv(self.config['intro']['n_filters'], (7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        # Feature extraction
        for i, n_layers in enumerate(self.config['n_layers']):
            if i == 0:
                x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
                x = SESkipBlock(
                    PreActResBlock(self.config['n_filters'][i], increase_dim=True)
                )(x, training=training)
            else:
                x = SESkipBlock(
                    PreActResBlock(self.config['n_filters'][i], downsample=True)
                )(x, training=training)
            for _ in range(n_layers - 1):
                x = SESkipBlock(
                    PreActResBlock(self.config['n_filters'][i])
                )(x, training=training)

        # Classification head
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)
        return y


class MobileNet(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, training: bool):
        alpha = self.config['alpha']
        # rho determines input size
        n_filters = self.config['initial_filters']

        # Intro
        x = nn.Conv(n_filters, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm()(x, training=training)
        x = nn.relu(x)
        n_filters = int(n_filters * alpha)
        x = MobileNetBlock(
            n_filters, down_sample=False, double_channel=True
        )
        n_filters *= 2

        # Feature extraction
        for n_layers in self.config['n_layers']:
            for i, n in enumerate(range(n_layers)):
                down_sample = (i == n_layers - 1)
                x = MobileNetBlock(
                    n_filters, down_sample=down_sample, double_channel=down_sample
                )(x, training=training)
                if down_sample:
                    n_filters *= 2
                else:
                    n_filters = int(n_filters * alpha)

        # Classification head
        x = nn.avg_pool(x, window_shape=(7, 7), strides=(1, 1), padding="VALID")
        x = x.squeeze(axis=(1, 2))
        x = nn.Dense(1000)(x)
        y = nn.softmax(x)
        return y
