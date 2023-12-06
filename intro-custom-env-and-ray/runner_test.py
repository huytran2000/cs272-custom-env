from environment_creation import GridWorldEnv
from ray.tune.registry import register_env

from ray.rllib.algorithms.algorithm import Algorithm


def env_creator(env_config):
    return GridWorldEnv()  # custom env


register_env("MyGrid", env_creator)

checkpoint_path = "/var/folders/l6/dkm7d0_s7_b882pgzbk6fyrm0000gn/T/tmpmn_u95w3"
algo = Algorithm.from_checkpoint(checkpoint_path)

env = GridWorldEnv(render_mode="human")

obs, _ = env.reset()

steps = 50
G = 0

for _ in range(steps):
    # action = algo.compute_single_action(obs)
    action = env.action_space.sample()
    obs, r, te, tr, _ = env.step(action)
    G += r

    if te or tr:
        print(G)
        G = 0
        obs, _ = env.reset()


env.close()
