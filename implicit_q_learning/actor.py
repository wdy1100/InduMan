from typing import Tuple
import jax
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey


def update(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    batch: Batch,
    temperature: float  # renamed from "temperature" for clarity (higher = more confident)
) -> Tuple[Model, InfoDict]:
    """
    Update actor using AWR-style weighted behavior cloning:
        L(π) = -E[ w(s,a) * log π(a|s) ]
    where w(s,a) = exp( β * (Q(s,a) - V(s)) )
    
    To avoid numerical overflow, we normalize the advantage before exponentiation.
    """
    # Compute Q and V
    v = value(batch.observations)  # (B,)
    q1, q2 = critic(batch.observations, batch.actions)  # (B,)
    q = jnp.minimum(q1, q2)  # (B,)

    # Advantage: A(s,a) = Q(s,a) - V(s)
    adv = q - v  # (B,)

    # Normalize advantage to stabilize exponentiation (optional but recommended)
    # This is equivalent to: exp(β * (adv - max(adv))) → prevents overflow
    normalized_adv = adv - jax.lax.stop_gradient(adv.max())
    exp_weights = jnp.exp(temperature * normalized_adv)

    # Optional: clip weights to prevent extreme values (e.g., > 100)
    exp_weights = jnp.minimum(exp_weights, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # Forward pass through actor
        dist = actor.apply(
            {'params': actor_params},
            batch.observations,
            rngs={'dropout': key}
        )
        log_probs = dist.log_prob(batch.actions)  # (B,)
        actor_loss = -(exp_weights * log_probs).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'advantage_mean': adv.mean(),
            'advantage_max': adv.max(),
            'weight_mean': exp_weights.mean(),
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info