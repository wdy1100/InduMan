import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple,Union
from functools import partial
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

# --- Type Aliases ---
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

Observation = Union[jnp.ndarray, Dict[str, jnp.ndarray]]

def flatten_observation(obs: dict) -> jnp.ndarray:
    """
    Concatenate the observations in dictionary form into a single vector.
    Suppose obs contains keys such as 'image1', 'image2', 'robot_state', etc.
    """
    if isinstance(obs, dict):
        # Order matters! Keep consistent across dataset and model.
        keys = ['image1', 'image2', 'robot_state']
        parts = [obs[k] for k in keys if k in obs]
        return jnp.concatenate(parts, axis=-1)
    else:
        return obs

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    if scale is None:
        scale = jnp.sqrt(2)
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

class Encoder(nn.Module):
    output_dim: int = 256  # make it configurable and larger

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Assume input is (B, H, W, C), normalized to [0,1]
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.LayerNorm()(x)
        return x

class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: Any
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
        x = jnp.mean(x, axis=(1, 2))
        # x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[1,1], block_cls=ResNetBlock)


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)