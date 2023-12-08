from env_creation2 import CarTrack1Env

env = CarTrack1Env(render_mode="human")

obs, _ = env.reset()
print(f'obs: {obs}')
print(f'goal_lane: {env.goal_lane_id}')

num_steps = 100

a_wait = [0]*100
a_one = 1
a_left = 4
a = [1, 7, 2, 2, 1, 3, 3]
a_optimal = [0, 3, 1, 2, 6, 1, 3, 3]
a1 = [0, 4, 0, 0, 5, 5, 6, 7, 1, 0, 4, 1, 6, 3, 3]
G = 0
for i in range(0):
    obs, r, te, tr, info = env.step(a_optimal[i])

    print(f'obs: {obs}')
    print(f'r: {r}')
    G += r

    if te or tr:
        print(f'info: {info}')
        print(f'G: {G}')
        break
        # obs, _ = env.reset()
        # G = 0

# check env randomizer
# for i in range(10):
#     _, _ = env.reset()
#     print(f'Goal Lane: {env.goal_lane_id}')

env.close()
