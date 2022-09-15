#  粒子群算法

import numpy as np
import matplotlib.pyplot as plt


N_update = 100
N_pop = 200
X_limit = [-5,5]
V_limit = (-0.4,0.4)
W = 0.9   # 惯性  
c1 = 0.8  # 学习效率 总群最优
c2 = 0.6  # 学习效率 个体最优 




def F(x):
    return  x*np.sin(2*x) #np.power(x,5) - np.power(x,4) - 10*np.power(x,3) + 6*x*x + 5*x



pop = np.zeros((N_pop,2))   # 200 * 2 的列表，存储200个粒子的位置，速度

#print(pop)



for i in range(N_pop):   # 初始化粒子位置,速度，各自适应度
    pop[i][0] = (np.random.random() - 0.5) * (X_limit[1] - X_limit[0])  # 位置
    pop[i][1] = (np.random.random() - 0.5) * (V_limit[1] - V_limit[0])  # 速度





G_best = []   # 存储每次更新的总群最优
P_best = []   # 存储每个个体的历史最优



def Gbest(pop):    #求解当前代的总群最优
    G_best = pop[0][0] 
    for i in range(N_pop):
        if F(pop[i][0]) > F(G_best):
            G_best = pop[i][0]
    return G_best
        
        
        
    

def update(pop,P_best):        # 先更新速度，后更新位置，再计算适应度
    G_best = Gbest(pop)
    for i in range(N_pop):
        
        pop[i][1] = W * pop[i][1] + np.random.random()*c1 * G_best + np.random.random()*c2 * P_best[i]
        if pop[i][1] > V_limit[1]:
            pop[i][1] = V_limit[1]
        elif pop[i][1] < V_limit[0]:
            pop[i][1] = V_limit[0]

        pop[i][0] += pop[i][1]
        if pop[i][0] > X_limit[1]:
            pop[i][0] = X_limit[1]
        elif pop[i][0] < X_limit[0]:
            pop[i][0] = X_limit[0]

    return pop



for i in range (N_pop):    #  初始化个体最佳位置
    P_best.append(pop[i][0])
    
   
for i in range(N_update):     # 运行部分
    
    pop = update(pop,P_best)
    
    G_best.append(Gbest(pop))
    

    for i in range(N_pop):      # 个体最优更新
        if F(pop[i][0]) > F(P_best[i]):
            P_best[i] = pop[i][0]


MAX = F(G_best[0])
MAX_x = G_best[0]
for i in range(N_update):
    if MAX < F(G_best[i]):
        MAX =F(G_best[i])
        MAX_x = G_best[i]
        
print("输出最大值",MAX)
print("输出最大值的位置",MAX_x )


fig  = plt.figure()

ax = fig.add_subplot(111)

x_fun = np.linspace(-5,5,200)

#print(x_fun)
y_fun = []

for i in range (200):
    y_value = F(x_fun[i])
    y_fun.append(y_value)
    

#ax.scatter(x_fun,y_fun,c = 'r',marker = 'o')   #绘制了函数图像散点图

plt.plot(x_fun,y_fun)    #绘制了函数图像
plt.show()    







