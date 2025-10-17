""" Define parameters for algorithms. """

import argparse


def str2bool(v):
    return v.lower() == "true"


def str2intlist(value):
    if not value:
        return value
    else:
        return [int(num) for num in value.split(",")]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(",")]


def create_parser():
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        "Robot Learning Algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # environment
    parser.add_argument(
        "--env", type=str, default="DUAL_HOLE_IN_PEG", help="environment name",
    )
    parser.add_argument("--env_type", type=str, default="assembly", help="environment type")
    parser.add_argument("--task_assets_path", type=str, default=None, help='Path to the task assets')
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_threads", type=int, default=2, help="Number of threads")
    parser.add_argument("--device",type=str,default="cuda",help="device to use")
    add_method_arguments(parser)

    return parser


def add_method_arguments(parser):
    # algorithm
    parser.add_argument(
        "--algo",
        type=str,
        default="bc",
        choices=["sac", "ppo", "ddpg", "td3", "bc", "gail", "dac",],
    )
    # InduMan
    parser.add_argument("--image_agmt", type=str2bool, default=False, help="use image agmt for InduMan")
    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--init_ckpt_path", type=str, default=None)

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument(
        "--num_eval", type=int, default=1, help="number of episodes for evaluation"
    )

    # environment
    try:
        parser.add_argument("--screen_width", type=int, default=480)
        parser.add_argument("--screen_height", type=int, default=480)
    except:
        pass
    parser.add_argument("--action_repeat", type=int, default=1)

    # misc
    parser.add_argument("--run_prefix", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")

    # log
    parser.add_argument("--average_info", type=str2bool, default=True)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--evaluate_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--log_root_dir", type=str, default="log")
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="set it True if you want to use wandb",
    )
    parser.add_argument("--wandb_entity", type=str, default="clvr")
    parser.add_argument("--wandb_project", type=str, default="robot-learning")
    parser.add_argument("--record_video", type=str2bool, default=False)
    parser.add_argument("--record_video_caption", type=str2bool, default=False)
    try:
        parser.add_argument("--record_demo", type=str2bool, default=False)
    except:
        pass

    # observation normalization
    parser.add_argument("--normalizer", type=str, default="bn")
    parser.add_argument("--ob_norm", type=str2bool, default=True)
    parser.add_argument("--max_ob_norm_step", type=int, default=int(1e8))
    parser.add_argument(
        "--clip_obs", type=float, default=200, help="the clip range of observation"
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=10,
        help="the clip range after normalization of observation",
    )

    parser.add_argument("--max_global_step", type=int, default=int(1e6))
    parser.add_argument(
        "--batch_size", type=int, default=128, help="the sample batch size"
    )

    add_policy_arguments(parser)

    # arguments specific to algorithms
    args, unparsed = parser.parse_known_args()
    if args.algo == "bc":
        add_il_arguments(parser)
        add_bc_arguments(parser)

    return parser


def add_policy_arguments(parser):
    # network
    parser.add_argument("--policy_mlp_dim", type=str2intlist, default=[128, 128])
    parser.add_argument("--critic_mlp_dim", type=str2intlist, default=[128, 128])
    parser.add_argument("--critic_ensemble", type=int, default=1)
    parser.add_argument(
        "--policy_activation", type=str, default="relu", choices=["relu", "elu", "tanh"]
    )
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--gaussian_policy", type=str2bool, default=True)

    # encoder
    parser.add_argument(
        "--encoder_type", type=str, default="mlp", choices=["mlp", "cnn","r3m", "resnet18"]
    )
    parser.add_argument("--finetune_encoder", type=bool, default=False)
    parser.add_argument("--resnet_in_r3m", type=str, default="resnet50")
    parser.add_argument("--encoder_image_size", type=int, default=128)
    parser.add_argument("--random_crop", type=str2bool, default=False)
    parser.add_argument("--encoder_conv_dim", type=int, default=32)
    parser.add_argument("--encoder_kernel_size", type=str2intlist, default=[3, 3, 3, 3])
    parser.add_argument("--encoder_stride", type=str2intlist, default=[2, 1, 1, 1])
    parser.add_argument("--encoder_conv_output_dim", type=int, default=50)
    parser.add_argument("--encoder_soft_update_weight", type=float, default=0.95)
    args, unparsed = parser.parse_known_args()
    if args.encoder_type == "cnn":
        parser.set_defaults(screen_width=100, screen_height=100)
        parser.set_defaults(policy_mlp_dim=[1024, 1024])
        parser.set_defaults(critic_mlp_dim=[1024, 1024])
        parser.add_argument("--asym_ac", type=str2bool, default=False)

    # actor-critic
    parser.add_argument(
        "--actor_lr", type=float, default=3e-4, help="the learning rate of the actor"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=3e-4, help="the learning rate of the critic"
    )
    parser.add_argument(
        "--critic_soft_update_weight",
        type=float,
        default=0.995,
        help="the average coefficient",
    )

    parser.add_argument("--log_std_min", type=float, default=-10)
    parser.add_argument("--log_std_max", type=float, default=2)

    # absorbing state
    parser.add_argument("--absorbing_state", type=str2bool, default=False)


def add_on_policy_arguments(parser):
    parser.add_argument("--rollout_length", type=int, default=2000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--advantage_norm", type=str2bool, default=True)


def add_off_policy_arguments(parser):
    parser.add_argument(
        "--buffer_size", type=int, default=int(1e6), help="the size of the buffer"
    )
    parser.set_defaults(warm_up_steps=1000)


def add_il_arguments(parser):
    parser.add_argument("--demo_path", type=str, default=None, help="path to demos")
    parser.add_argument("--demo_low_level", type=str2bool, default=False, help="use low level actions for training")
    parser.add_argument(
        "--demo_subsample_interval",
        type=int,
        default=1,
        # default=20, # used in GAIL
        help="subsample interval of expert transitions",
    )
    parser.add_argument(
        "--demo_sample_range_start", type=float, default=0.0, help="sample demo range"
    )
    parser.add_argument(
        "--demo_sample_range_end", type=float, default=1.0, help="sample demo range"
    )


def add_bc_arguments(parser):
    parser.set_defaults(gaussian_policy=False)
    parser.set_defaults(max_global_step=500)
    parser.set_defaults(evaluate_interval=100)
    parser.set_defaults(ob_norm=False)
    parser.add_argument(
        "--bc_lr", type=float, default=1e-3, help="learning rate for bc"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0,
        help="how much of dataset to leave for validation set",
    )


def argparser():
    """ Directly parses the arguments. """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    return args, unparsed
