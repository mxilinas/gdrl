from ray.air.config import RunConfig
from godot_rl.core.godot_env import GodotEnv, os
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.train import CheckpointConfig
from ray.tune.registry import register_env
import ray
from ray import tune
import yaml
import argparse as ap


def get_cli_args():
    """Get command line arguments."""
    parser = ap.ArgumentParser()
    parser.add_argument("--config_file", default="pre-train.yaml")
    parser.add_argument("--log_dir", default="logs")
    return parser.parse_args()


def get_experiment_config(args):
    """Load the experiment configuration settings from the yaml file."""
    with open(args.config_file) as f:
        return yaml.safe_load(f)


def get_policy_names(exp):
    """Get the policy names from the godot environment."""
    print("Starting a temporary environment to get policy names")
    tmp_env = GDRLPettingZooEnv(
        config={
            "env_path": exp["config"]["env_config"]["env_path"],
            "show_window": False
        }
    )
    policy_names = tmp_env.agent_policy_names
    print("Closing the temporary environment.")
    tmp_env.close()
    return policy_names


def setup_multiagent_config(exp, policy_names):
    """Add multiagent config settings to the configuration file."""
    exp["config"]["multiagent"] = {
        "policies": {policy_name: PolicySpec() for policy_name in policy_names},
        "policy_mapping_fn": policy_mapping_fn,
    }


if __name__ == "__main__":

    args = get_cli_args()
    exp = get_experiment_config(args)

    policy_names = get_policy_names(exp)

    def policy_mapping_fn(agent_id: int, episode, worker, **kwargs) -> str:
        return policy_names[agent_id]

    def env_creator(env_config):
        index = env_config.worker_index * exp["config"]["num_envs_per_runner"]
        index += env_config.vector_index
        gdrl_env = GDRLPettingZooEnv(
            config=env_config,
            port=index + GodotEnv.DEFAULT_PORT,
            seed=index
        )
        return ParallelPettingZooEnv(gdrl_env)

    ray.init(runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}})

    register_env(exp["config"]["env"], env_creator)

    setup_multiagent_config(exp, policy_names)

    tuner = tune.Tuner(
        trainable=exp["algorithm"],
        param_space=exp["config"],
        run_config=RunConfig(
            storage_path=os.path.abspath(args.log_dir),
            stop=exp["stop"],
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=exp["checkpoint_frequency"]
            )
        ),
    )

    result = tuner.fit()
