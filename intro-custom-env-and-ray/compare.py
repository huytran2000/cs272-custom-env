from env_creation2 import CarTrack1Env
from ray.tune.registry import register_env

from ray.rllib.algorithms.algorithm import Algorithm


def env_creator(env_config):
    return CarTrack1Env()  # custom env


register_env("MyTrack", env_creator)

# checkpoint_path = "/var/folders/l6/dkm7d0_s7_b882pgzbk6fyrm0000gn/T/tmp6047c88a"
# checkpoint_path = "/var/folders/l6/dkm7d0_s7_b882pgzbk6fyrm0000gn/T/tmp3s3osq72"
checkpoint_path = "./cp_log/tmpwcbgi0ht"
# 20 eps + 20
algo = Algorithm.from_checkpoint(checkpoint_path)

env = CarTrack1Env()  # render_mode="human")

obs, _ = env.reset()
print(f'Goal_Lane: {env.goal_lane_id}')
env._render_frame()
# print("Agent actions: ", end="")

steps = 200
G = 0

for _ in range(steps):
    action = algo.compute_single_action(obs)
    # print(f'{action} ', end="")

    obs, r, te, tr, info = env.step(action)
    # env._render_frame()
    G += r

    if te or tr:
        print()
        print(f'Agent G: {G}')
        print(f'{info}')
        G = 0
        print("----------------------------------------------")
        break

obs, _ = env.reset()
print(f'Goal_Lane: {env.goal_lane_id}')
# env._render_frame()
# print("Agent actions: ", end="")

for _ in range(steps):
    action = env.action_space.sample()

    obs, r, te, tr, info = env.step(action)
    G += r

    if te or tr:
        print(f'Agent G: {G}')
        print(f'{info}')
        G = 0
        print("----------------------------------------------")
        break


env.close()
