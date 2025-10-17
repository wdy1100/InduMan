from typing import Tuple
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params

def expectile_loss(diff: jnp.ndarray, expectile: float = 0.8) -> jnp.ndarray:
    """
    Compute expectile loss: 
        L_τ(u) = |τ - 1_{u < 0}| * u²
    """
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def update_v(
    critic: Model,
    value: Model,
    batch: Batch,
    expectile: float
) -> Tuple[Model, InfoDict]:
    """
    Update value network V(s) by minimizing expectile regression loss:
        L = 𝔼[ L_τ(Q(s,a) - V(s)) ]
    """
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        value_loss = expectile_loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'q_minus_v_mean': (q - v).mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)
    return new_value, info


def update_q(
    critic: Model,
    target_value: Model,
    batch: Batch,
    discount: float
) -> Tuple[Model, InfoDict]:
    """
    Update Q-networks using TD error with V(s') as target (IQL style):
        y = r + γ * (1 - done) * V(s')
        L = (Q1(s,a) - y)² + (Q2(s,a) - y)²
    """
    next_v = target_value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(
            {'params': critic_params},
            batch.observations,
            batch.actions
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_mean': q1.mean(),
            'q2_mean': q2.mean(),
            'target_q_mean': target_q.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    return new_critic, info