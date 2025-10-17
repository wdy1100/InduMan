import os
from typing import Tuple
import gymnasium as gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import yaml
import torch
import jax

import wrappers
from dataset_utils import InduManDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
from common import Batch, flatten_observation

from isaacsim import SimulationApp

from InduMan.utils import get_task_cfg_path,get_assets_path

"""
 code for training offline
 not use encoder
python InduMan/implicit_q_learning/train_offline.py --data_path=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/iql_data/pkl/DUAL_HOLE_IN_PEG.pkl --save_dir=/home/wdy02/wdy_program/simulation_plus/IsaacLab/InduMan/record_data/iql_data --config=/home/wdy02/wdy_program/simulation_plus/IsaacLab/InduMan/implicit_q_learning/configs/default.py
"""
FLAGS = flags.FLAGS
flags.DEFINE_string('env_type', 'assembly', 'Environment type.')
flags.DEFINE_string('env_name', 'DUAL_HOLE_IN_PEG', 'Environment name.')
flags.DEFINE_bool('task_assets_path', None, 'Task assets path.')
flags.DEFINE_string('data_path', None, 'dataset path.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_bool('use_encoder', False, 'Use ResNet18 for the image encoder.')
flags.DEFINE_string('encoder_type', '', 'Type of image encoder.(vip or r3m)')
flags.DEFINE_integer('seed', 123, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('ckpt_interval', 2000, 'Checkpoint interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e5), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_bool('headless', True, 'Run in headless mode.')
config_flags.DEFINE_config_file(
    'config',
    'config/default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

simulation_app = SimulationApp({"headless": FLAGS.headless})

def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         env_type: str,
                         data_path: str,
                         task_assets_path,
                         seed: int,
                         use_encoder=False,
                         encoder_type = '',
                         ) -> Tuple[gym.Env, InduManDataset]:
    if task_assets_path is None:
        task_assets_path = get_assets_path()

    env_cfg_path = os.path.join(get_task_cfg_path(), f"{env_name}.yaml")
    with open(env_cfg_path, 'r') as f:
        task_config = yaml.safe_load(f)
    if env_type == "assembly":
        from InduMan.scripts import FrankaAssemblyGym
        env = FrankaAssemblyGym(is_replay=False,
                        max_episode_steps=task_config['max_episode_steps'],
                      stage_units=task_config['stage_units'],
                      image_size=task_config['image_size'],
                      physics_dt=task_config['physics_dt'],
                      rendering_dt=task_config['rendering_dt'],
                      table_height=task_config['table_height'],
                      task_name=env_name,
                      task_message=task_config['task_message'],
                      objects_to_manipulate=task_config['objects_to_manipulate'],
                      objects_to_interact=task_config['objects_to_interact'],
                      support = task_config['support'],
                      task_assets_path = task_assets_path,
                      gripper_opened_position = task_config['gripper_opened_position'],
                      )
    elif env_type == "disassembly":
        from InduMan.scripts import FrankaDisassemblyGym
        env = FrankaDisassemblyGym(is_replay=False,
                            max_episode_steps=task_config['max_episode_steps'],
                        stage_units=task_config['stage_units'],
                        image_size=task_config['image_size'],
                        physics_dt=task_config['physics_dt'],
                        rendering_dt=task_config['rendering_dt'],
                        table_height=task_config['table_height'],
                        task_name=env_name,
                        task_message=task_config['task_message'],
                        objects_to_manipulate=task_config['objects_to_manipulate'],
                        objects_to_interact=task_config['objects_to_interact'],
                        task_assets_path=task_assets_path,
                        gripper_opened_position=task_config['gripper_opened_position'],
                        )
    else:
        raise ValueError(f"Unsupported task type: {env_type}")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # env = wrappers.EpisodeMonitor(env)
    # env = wrappers.SinglePrecision(env)

    dataset = InduManDataset(data_path=data_path)

    return env, dataset


def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    tb_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, 'tb', str(FLAGS.seed))
    ckpt_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, 'ckpts', str(FLAGS.seed))

    summary_writer = SummaryWriter(tb_dir, write_to_disk=True)

    env, dataset = make_env_and_dataset(env_name=FLAGS.env_name, 
                                        env_type=FLAGS.env_type, 
                                        data_path=FLAGS.data_path,
                                        task_assets_path=FLAGS.task_assets_path,
                                        seed=FLAGS.seed,
                                        use_encoder=FLAGS.use_encoder,
                                        encoder_type=FLAGS.encoder_type,)

    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    if FLAGS.use_encoder:
        sample_obs = dataset.observations[0]  # dict with 'image1', 'image2', 'robot_state'
        init_obs = flatten_observation(sample_obs)
    else:
        init_obs = env.observation_space.sample()
    agent = Learner(FLAGS.seed,
                    init_obs,
                    env.action_space.sample(),
                    max_steps=FLAGS.max_steps,
                    use_encoder=FLAGS.use_encoder,
                    **kwargs)

    encoder_model = None
    if FLAGS.use_encoder:
        if FLAGS.encoder_type == 'r3m':
            from r3m import load_r3m
            encoder_model = load_r3m("resnet50")
            encoder_model.eval()
            for p in encoder_model.parameters():
                p.requires_grad = False

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                    smoothing=0.1,
                    disable=not FLAGS.tqdm):
        env.world.step(render=True)
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent=agent, 
                                  env=env, 
                                  num_episodes=FLAGS.eval_episodes, 
                                  use_encoder=FLAGS.use_encoder, 
                                  encoder_type=FLAGS.encoder_type,
                                  encoder_model=encoder_model)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                    eval_returns,
                    fmt=['%d', '%.1f'])
            
        if i % FLAGS.ckpt_interval == 0:
            agent.save(ckpt_dir, i)

    if not i % FLAGS.ckpt_interval == 0:
        # Save last step if it is not saved.
        agent.save(ckpt_dir, i)
    print('--------Training complete.--------')

    # Close the simulation app to avoid post_quit() error
    simulation_app.close()

if __name__ == '__main__':
    app.run(main)
