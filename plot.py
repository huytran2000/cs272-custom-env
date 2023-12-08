import csv
from matplotlib import pyplot as plt

# file = open(
#     "/Users/huy/ray_results/DQN_MyTrack_2023-12-07_22-58-36ui_hap5n/progress.csv")
# csvreader = csv.reader(file)
# erm = []

# next(csvreader)  # get header

# # get rows
# for row in csvreader:
#     erm.append(float(row[2]))

# # print(erm)

# x = [i+1 for i in range(100)]
# plt.plot(x, erm)
# plt.ylabel("Episode Mean Return")
# plt.xlabel("Iteration ID")
# plt.title("Mean Return Over Iteration Of 2nd Trained Agent")

# plt.show()

r_optimal = [194, 194, 193, 194, 194, 194, 194, 194, 192, 194]
r_agent = [-217, 193, 192, 194, -220, 190, -277, 192, 192, 194]
r_rand = [-200, -203, -200, -201, -202, -201, -203, -202, -200, -201]

x = [i+1 for i in range(10)]
plt.scatter(x, r_optimal, label="Optimal")
plt.scatter(x, r_agent, label="Agent")
plt.scatter(x, r_rand, label="Random Action")
plt.ylabel("Episode's Return")
plt.xlabel("Env ID")
plt.title("Comparison of Returns Over Sampled Environments For Agents")
plt.xticks(x)
plt.legend()

plt.show()
