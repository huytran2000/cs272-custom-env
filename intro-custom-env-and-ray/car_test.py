from environment_creation import CarTrackEnv
from ray.tune.registry import register_env

from ray.rllib.algorithms.algorithm import Algorithm


def env_creator(env_config):
    return CarTrackEnv()  # custom env


register_env("MyTrack", env_creator)

# checkpoint_path = "/var/folders/l6/dkm7d0_s7_b882pgzbk6fyrm0000gn/T/tmp6047c88a"
checkpoint_path = "/var/folders/l6/dkm7d0_s7_b882pgzbk6fyrm0000gn/T/tmp10wg8emt"
# /var/folders/l6/dkm7d0_s7_b882pgzbk6fyrm0000gn/T/tmpxmdcb6ag
algo = Algorithm.from_checkpoint(checkpoint_path)

env = CarTrackEnv()  # render_mode="human")

obs, _ = env.reset()
env._render_frame()

steps = 50
G = 0

for _ in range(steps):
    action = algo.compute_single_action(obs)
    print(f'{action} ', end="")
    # action = env.action_space.sample()
    obs, r, te, tr, _ = env.step(action)
    G += r

    if te or tr:
        print()
        print(f'G: {G}')
        G = 0
        obs, _ = env.reset()

print()
print(f'Final G: {G}')
env.close()
