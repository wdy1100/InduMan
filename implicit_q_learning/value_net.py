from typing import Callable, Sequence, Tuple, Dict, Union
import jax.numpy as jnp
from flax import linen as nn

from common import MLP, Observation


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, observations: Observation, actions: jnp.ndarray) -> jnp.ndarray:
        # Standardize action shape
        if actions.ndim == 3:
            actions = jnp.squeeze(actions, axis=1)

        inputs = jnp.concatenate([observations, actions], axis=-1)
        q_value = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(q_value, axis=-1)

class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, observations: Observation, actions: jnp.ndarray):
        q1 = Critic(self.hidden_dims, self.activations)(observations, actions)
        q2 = Critic(self.hidden_dims, self.activations)(observations, actions)
        return q1, q2
    
# ValueCritic similarly
class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: Observation) -> jnp.ndarray:
        v = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(v, axis=-1)