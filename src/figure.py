import matplotlib.pyplot as plt
import numpy as np

# 假数据
avg_reward = [722.8, 399.5]  # LLM-Agent 和 Baseline-Agent 的平均奖励
avg_step = [497.0, 334.0]  # LLM-Agent 和 Baseline-Agent 的平均步数

bar_width = 0.35
index = np.arange(len(avg_reward))

fig, ax1 = plt.subplots()

# 绘制 avg_reward 的柱状图
rects1 = ax1.bar(index, avg_reward, bar_width, label='Average Reward', color='b')

# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制 avg_step 的柱状图
rects2 = ax2.bar(index + bar_width, avg_step, bar_width, label='Average Step', color='r')

# 设置标题和坐标轴标签
plt.title('Risk-hard Environment')
ax1.set_xlabel('Agent')
ax1.set_ylabel('Average Reward', color='b')
ax2.set_ylabel('Average Step', color='r')

# 设置横坐标标签
plt.xticks(index + bar_width/2, ['LLM-Agent', 'Baseline-Agent'])

# 显示图例
fig.legend(loc='upper right')

plt.savefig("../doc/risk_hard.png")
