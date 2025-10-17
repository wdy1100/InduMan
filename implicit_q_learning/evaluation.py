from typing import Dict, Optional
import torch
from torchvision import transforms, models
import flax.linen as nn
import gymnasium as gym
import numpy as np
from r3m import load_r3m


def evaluate(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    max_episode_steps: int = 1000,
    use_encoder: bool = False,
    encoder_type: str = '',
    encoder_model: Optional[torch.nn.Module] = None,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate the agent in the environment.

    Args:
        agent: Flax policy (expects dict observation if use_encoder=True)
        env: Gymnasium environment
        num_episodes: number of episodes to evaluate
        max_episode_steps: max steps per episode
        use_encoder: whether to encode images
        encoder_type: 'r3m' or 'resnet' (only used if encoder_model is None)
        encoder_model: pre-loaded encoder (recommended)
        device: device for encoder (e.g., 'cuda')
    """
    stats = {'return': [], 'length': []}

    # Load encoder once if not provided
    if use_encoder and encoder_model is None:
        if encoder_type == 'r3m':
            encoder_model = load_r3m("resnet50").to(device).eval()
            for p in encoder_model.parameters():
                p.requires_grad = False
        elif encoder_type == 'resnet':
            resnet = models.resnet50(pretrained=True)
            encoder_model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
            for p in encoder_model.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    for _ in range(num_episodes):
        observation = env.reset()
        obs = _convert_observation(
            observation, 
            use_encoder=use_encoder, 
            encoder_model=encoder_model, 
            encoder_type=encoder_type,
            device=device
        )
        total_reward = 0.0
        step = 0

        for step in range(max_episode_steps):
            action = agent.sample_actions(obs, temperature=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                break

            obs = _convert_observation(
                observation,
                use_encoder=use_encoder,
                encoder_model=encoder_model,
                encoder_type=encoder_type,
                device=device
            )

        stats['return'].append(total_reward)
        stats['length'].append(step + 1)

    # Compute mean
    for k in stats:
        stats[k] = np.mean(stats[k])

    return stats


def _convert_observation(
    ob: dict,
    use_encoder: bool = False,
    encoder_model: Optional[torch.nn.Module] = None,
    encoder_type: str = '',
    device: str = 'cuda'
):
    # === Extract robot state ===
    robot_state_keys = [
        "ee_pose",
        "gripper_state",
        "joint_positions",
        "joint_velocities",
        "joint_forces_torques"
    ]
    robot_state = []
    for key in robot_state_keys:
        for k, v in ob.items():
            if key in k:
                robot_state.append(v)
    robot_state = np.concatenate(robot_state, axis=0).astype(np.float32)

    if not use_encoder:
        # Return dict for consistency (even without images)
        return robot_state

    # === Preprocess image: HWC uint8 [0,255] -> CHW float [0,1] ===
    def preprocess(img):
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:  # HWC
            img = np.transpose(img, (2, 0, 1))  # -> CHW
        return torch.from_numpy(img).to(device)

    # === Encode images ===
    with torch.no_grad():
        img1 = preprocess(ob['agent_rgb'])
        img2 = preprocess(ob['hand_rgb'])

        if encoder_type == 'resnet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            img1 = normalize(img1)
            img2 = normalize(img2)

        feat1 = encoder_model(img1.unsqueeze(0)).squeeze().cpu().numpy()
        feat2 = encoder_model(img2.unsqueeze(0)).squeeze().cpu().numpy()

    new_obs = np.concatenate([feat1, feat2, robot_state], axis=0).astype(np.float32)
    return new_obs