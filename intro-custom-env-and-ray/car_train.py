from environment_creation import CarTrackEnv

from ray.tune.registry import register_env

from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
from ray.tune.logger import pretty_print


def env_creator(env_config):
    return CarTrackEnv()  # custom env


# register the name of the custom env
register_env("MyTrack", env_creator)

# getting the config dict
# set the environemnt; they can even do hyperparaameter tuning, u can set that; this is where u set the exploration aand stuff
config = DQNConfig()
config = config.environment(env="MyTrack")

# see the config algo uses; u can see from exploration_config, that they decreses epsilon overtime since agent should explroe less the more it learns
# nothing shows for env_config, but for num_rollout_workers, u see 0; if u set this to 1 or 2; then more workers will collect experience and put into replaay buffer
print('--------------')
print(config.env_config)
print(config.exploration_config)
print(config.num_rollout_workers)
print('--------------')

algo = DQN(config=config)

for i in range(15):
    result = algo.train()
    # print(pretty_print(result))
    print(f'Done -{i}-')

    if (i-1) % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
