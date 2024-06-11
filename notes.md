## Experimental steps

1. Pre-train two agents using adversarial learning.
    - Build and export godot environment.
    - Register environment with rllib.
    - Configure and train policies.

2. One group interacts with an agent without shaping. Another group interacts 
   with an agent with shaping.
   - Initialize a new experiment with the policy weights of a pre-trained agent.
   - One experiment involves configuring a new algorithm for training.
   - The other is just evaluation.

# RL

An agent interacts with an environment and receives rewards.
environments are defined with libaraies like PettingZoo or Gymnasium.

## General
done flags indicate the boundaries of episodes.
reset returns actions, states, rewards.

# rllib

## Core Concepts

The Tune lib allows you to manage algorithms 

num_envs_runners = parralelism

episodes are called rollouts.

given an environment and a policy, rollout workers produce batches of 
experience.

gather experience -> improve policy -> gather experience

all data takes the form of sample batches. Batches encode fragments of 
a trajectory.

trajectory includes:
actions, obs, rewards, dones, etc.

rollout_fragment_length = fragments (batches) per rollout worker.
train_batch_size = total fragments per batch.


1. Create environment.
    GDRLPettingZoo wrapper
    AlgorithmConfig.environment
    Register env with tune.
2. Specify and build algo.
    Create an instance of the algorithm object. PPOConfig for example.
        https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms
2. Train.
3. Save the result.

## Environments

The simulation of the world in which the agent solves a problem.

Custom environment classes must take a single env_config param.
Alternatively you can use a function that takes an env_config object and returns 
an environment instance.

Gymnasium is used for single-agent envs. PettingZoo is used for multi-agent 
envs.

Training speed is often limited by policy evaluation (produce exp) between 
steps. You can work around this by creating multiple envs and batching 
evaluations across them.

num_envs_per_runner = number of envs per worker (process).
num_envs_runners = number of workers (different processes)

### External envs

Has their own control thread.
self.get_action()
self.log_returns()

off policy decisions for human control? DQN
