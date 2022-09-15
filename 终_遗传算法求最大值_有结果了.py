#求函数 y = x^5 - x^4 -10*x^3 + 6*x^2 + 5x 在[-100,100]的最大值   --遗传算法

import numpy as np
import math
import random
import matplotlib.pyplot as plt

N_popnum = 200
N_dna = 20     # dna数为20
N_circlenum = 100
P_crossover = 0.7

P_mutate = 0.004
X_limit = [-5,5]






def F(x):          #获得函数值
    return x*np.sin(2*x)#np.power(x,5) - np.power(x,4) - 10*np.power(x,3) + 6*x*x + 5*x

def jiema(geti):
    x = geti.dot(2**np.arange(N_dna)[::-1]) / float(2 ** N_dna-1) * (X_limit[1] - X_limit[0]) + X_limit[0]
    return  x

def crossover(X):
    new_pop = []
    for child_fa in X:
        child = child_fa
        if random.random() < P_crossover:          # 交配概率为P_crossover
            n_ma = random.randint(0,N_popnum-1)
            child_ma = []
            child_ma = X[n_ma]       # 随机选择母代
            crs_position = random.randint(0,N_dna)           # 交叉位置随机
            for i in range(crs_position,N_dna):
                child[i] = child_ma[i]
            if F(jiema(child)) > F(jiema(child_fa)):       # 交叉后孩子的值高于父代，则以0.8 的概率孩子把父代替换    #选择
                if random.random() < 0.8:        
                    child = child
                else:
                    child = child_fa
            else:                          #若子代值小于父代，则以0.2 的概率保留子代
                if random.random() < 0.2:
                    child = child
                else:
                    child = child_fa

        new_pop.append(child)           #   新的个体加入

    return new_pop


def mutate(new_pop):
    for i in range(N_popnum):         # 随机产生变异
        if random.random() < P_mutate:
            x = new_pop[i]
            r = random.randint(0,N_dna-1)
            if x[r] < 1:
                x[r] = 1
            else:
                x[r] = 0
            new_pop[i] = x
    return new_pop
            
                
            
    


#def translateDNA():
X = np.random.randint(2,size=(N_popnum,N_dna))  # 初始化

# 这里N_dna*2的话，太大了，会溢出，提前出现负数

zuidazhi = []
for k in range(N_circlenum):
    X = crossover(X)
    X = mutate(X)
    t = F(jiema(X[0]))
    for i in range(N_popnum):
        if F(jiema(X[i]))>t:
            t = F(jiema(X[i]))
    zuidazhi.append(t)

MAX = max(zuidazhi)

print("函数在规定范围最大值为",MAX)
        

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


