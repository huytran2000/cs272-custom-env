from environment_creation import CarTrackEnv

env = CarTrackEnv(render_mode="human")

obs, _ = env.reset()
print(f'obs: {obs}')

num_steps = 100

a_wait = [0]*100
a = [0]*5 + [3, 1, 2, 2, 1, 3, 3]
a_optimal = [0, 3, 1, 2, 6, 1, 3, 3]
G = 0
for i in range(10):
    obs, r, te, tr, info = env.step(a_wait[i])

    print(f'obs: {obs}')
    print(f'r: {r}')
    G += r

    if te or tr:
        print(f'info: {info}')
        print(f'G: {G}')
        break
        # obs, _ = env.reset()
        # G = 0

env.close()
