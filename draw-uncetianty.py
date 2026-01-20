import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import random

# 1. 加载数据 (假设你的文件名为 data.csv)
# df = pd.read_csv('data.csv')
gam1, gam2 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0], [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]
gamma1, gamma2, MAE = [], [], []
combinations = list(itertools.product(gam1, gam2))
for cmd in combinations:
    gamma1.append(cmd[0])
    gamma2.append(cmd[1])


for i in range(len(gamma1)):
    mae = 7.27 + round(random.uniform(0, 3),2)
    MAE.append(mae)
# 模拟数据 (用于演示)
data = {
    'gamma_1': gamma1,
    'gamma_2': gamma2,
    'MAE': MAE
}
df = pd.DataFrame(data)



# 2. 将数据转换为矩阵格式 (Pivot table)
# 这一步是将长表转换为以 gamma_1 为行，gamma_2 为列，MAE 为值的矩阵
pivot_df = df.pivot(index="gamma_1", columns="gamma_2", values="MAE")

# 3. 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_df, annot=False, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'MAE'})

plt.title('MAE Variation with Hyperparameters $\gamma_1$ and $\gamma_2$')
plt.xlabel('$\gamma_2$')
plt.ylabel('$\gamma_1$')
plt.show()