from environment_creation import GridWorldEnv
import gymnasium as gym
import numpy as np

env = GridWorldEnv(render_mode="human")
# env = gym.make("CartPole-v1")

obs, _ = env.reset()

num_steps = 100

for _ in range(100):
    # no RL agent, here is just random actions
    a = env.action_space.sample()

    # print(env.observation_space.sample())
    # print(obs)
    obs, r, terminated, truncated, _ = env.step(a)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()


# from ray.tune.registry import register_env

# from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
# from ray.tune.logger import pretty_print


# def env_creator(env_config):
#     return GridWorldEnv()  # custom env


# # register the name of the custom env
# register_env("MyGrid", env_creator)

# # getting the config dict
# # set the environemnt; they can even do hyperparaameter tuning, u can set that; this is where u set the exploration aand stuff
# config = DQNConfig()
# config = config.environment(env="MyGrid")

# # see the config algo uses; u can see from exploration_config, that they decreses epsilon overtime since agent should explroe less the more it learns
# # nothing shows for env_config, but for num_rollout_workers, u see 0; if u set this to 1 or 2; then more workers will collect experience and put into replaay buffer
# print('--------------')
# print(config.env_config)
# print(config.exploration_config)
# print(config.num_rollout_workers)
# print('--------------')

# algo = DQN(config=config)

# for i in range(10):
#     result = algo.train()
#     print(pretty_print(result))

#     if i % 9 == 0:
#         checkpoint_dir = algo.save().checkpoint.path
#         print(f"Checkpoint saved in directory {checkpoint_dir}")
