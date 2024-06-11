import yaml
import ray
from ray.rllib.policy.policy import Policy
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv
from godot_rl.core.godot_env import GodotEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from argparse import ArgumentParser
PYTHONWARNINGS = "ignore::DeprecationWarning"


def get_policy_weights(policy_dict, name):
    if isinstance(policy_dict, dict):
        return policy_dict[name].get_weights()
    else:
        return policy_dict.get_weights()


def get_cli_args():
    """Get command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--config_file", default="shaping.yaml")
    parser.add_argument("--log_dir", default="logs")
    return parser.parse_args()


def get_experiment_config(args):
    """Load the experiment configuration settings from the yaml file."""
    with open(args.config_file) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":

    args = get_cli_args()
    exp = get_experiment_config(args)

    def env_creator(env_config):
        index = env_config.worker_index * exp["config"]["num_envs_per_runner"]
        index += env_config.vector_index
        return RayVectorGodotEnv(
            config=exp["config"]["env_config"],
            port=index + GodotEnv.DEFAULT_PORT,
            seed=index
        )

    register_env("shape", env_creator)

    policy = Policy.from_checkpoint(args.checkpoint_path)
    weights = get_policy_weights(policy, exp["policy_name"])
    weights = {'default_policy': weights}

    user_config = exp["config"]

    config = (
        PPOConfig()
        .environment("shape")
        .env_runners(num_env_runners=1)
        .training(
            lr=user_config["lr"],
            train_batch_size=user_config["train_batch_size"],
            model=user_config["model"],
        )
    )
    config.rollout_fragment_length = user_config["rollout_fragment_length"]
    config.num_env_runners = user_config["num_env_runners"]
    config.policies_to_train = ["default_policy"]

    ppo = config.build()
    ppo.set_weights(weights)

    for i in range(100):
        ppo.train()
