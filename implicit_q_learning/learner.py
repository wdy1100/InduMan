"""Implementation of Implicit Q-Learning (IQL) for continuous control."""

from typing import Optional, Sequence, Tuple
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    """
    Soft update of the target network parameters.
    θ_target = τ*θ + (1-τ)*θ_target
    """
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        critic.params,
        target_critic.params
    )
    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    """
    Perform one step of IQL update:
      1. Update value network V(s) via expectile regression.
      2. Update actor via AWR-style weighted behavior cloning.
      3. Update critic Q(s,a) via TD error with V(s') as target.
      4. Soft-update target critic.
    """
    # Step 1: Update value function V(s)
    new_value, value_info = update_v(target_critic, value, batch, expectile)

    # Step 2: Update actor using AWR-style objective
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(
        key, actor, target_critic, new_value, batch, temperature
    )

    # Step 3: Update critic Q(s,a)
    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    # Step 4: Soft update target critic
    new_target_critic = target_update(new_critic, target_critic, tau)

    # Merge logging info
    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 use_encoder: bool =False):
        """
        An implementation of Implicit Q-Learning (IQL): 
        https://arxiv.org/abs/2110.06169
        
        Note: This is NOT Soft Actor-Critic (SAC). IQL is an offline RL algorithm 
        that learns a value function and uses expectile regression + AWR-style actor update.
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)

        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False,
                                            use_encoder=use_encoder)

        # Actor optimizer with optional cosine decay
        if opt_decay_schedule == "cosine":
            if max_steps is None:
                raise ValueError("max_steps must be provided when using cosine decay.")
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_optimiser = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn)
            )
        else:
            actor_optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=actor_optimiser
        )

        # Critic and value networks
        critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr)
        )

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr)
        )

        # Target critic (no optimizer)
        target_critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions]
        )

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Sample actions from the policy and clip to [-1, 1].
        """
        rng, actions = policy.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature=temperature
        )
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1.0, 1.0)

    def update(self, batch: Batch) -> InfoDict:
        """
        Perform one training step and return logging info.
        Important: Do NOT modify self.rng during MSE computation to avoid RNG pollution.
        """
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_critic,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.temperature
        )

        # Compute MSE between behavior actions and current policy (for logging only)
        # Use new_rng and new_actor.params to avoid side effects
        _, sampled_actions = policy.sample_actions(
            new_rng,
            self.actor.apply_fn,
            new_actor.params,  # use updated policy
            batch.observations,
            temperature=0.0  # deterministic evaluation
        )
        sampled_actions = jnp.clip(sampled_actions, -1.0, 1.0)
        info['mse'] = jnp.mean((batch.actions - sampled_actions) ** 2)

        # Update internal state
        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info

    def save(self, ckpt_dir: str, step: int):
        """Save model checkpoints using safe path joining."""
        os.makedirs(ckpt_dir, exist_ok=True)
        self.actor.save(os.path.join(ckpt_dir, f"{step}_actor"))
        self.critic.save(os.path.join(ckpt_dir, f"{step}_critic"))
        self.target_critic.save(os.path.join(ckpt_dir, f"{step}_target_critic"))
        self.value.save(os.path.join(ckpt_dir, f"{step}_value"))

    def load(self, ckpt_dir: str, step: int):
        """Load model checkpoints."""
        self.actor = self.actor.load(os.path.join(ckpt_dir, f"{step}_actor"))
        self.critic = self.critic.load(os.path.join(ckpt_dir, f"{step}_critic"))
        self.target_critic = self.target_critic.load(os.path.join(ckpt_dir, f"{step}_target_critic"))
        self.value = self.value.load(os.path.join(ckpt_dir, f"{step}_value"))