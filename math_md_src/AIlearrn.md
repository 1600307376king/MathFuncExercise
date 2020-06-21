[TOC]

1、梯度下降算法

2、正向传播

3、反向传播

# 一、梯度下降算法

#### 			梯度下降法时一种深度学习优化算法

1. 公式表述：

​		 （1）数学公式：
$$
y = f(x)
$$
​		（2）关于函数权重wi求偏导：
$$
\frac{\partial y}{\partial w_i} = \frac{\partial f(x)}{\partial w_i}
$$
​		（3）例：
$$
f(x) = 2x^2
$$
​		（4）f(x)关于x求偏导：
$$
\frac{\partial f(x)}{\partial x} = 4x
$$
​		（5）给定x初始值x=6，学习率为η=0.1，偏导值为2.4，更新x之继续迭代知道x趋于4x=0，即x=0
$$
x\rightarrow x-\eta\ast4x
$$
​		（6）图示：



2. 梯度下降法可能存在的问题

​		（1）局部最小值

<img src="C:\Users\jjc\PycharmProjects\MathFuncExercise\math_src\local_min_val.png" alt="local_min_val" style="zoom:50%;" />

​		（2）梯度消失

<img src="C:\Users\jjc\PycharmProjects\MathFuncExercise\math_src\gradient_disappears.PNG" alt="gradient_disappears" style="zoom:50%;" />

​		（3）梯度爆炸

<img src="C:\Users\jjc\PycharmProjects\MathFuncExercise\math_src\gradient_explosion.png" alt="gradient_explosion" style="zoom:50%;" />