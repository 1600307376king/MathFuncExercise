[TOC]

1、梯度下降算法

2、正向传播

3、反向传播

# 第一部分 数学基础

## 第一章 线性代数

1. 矩阵计算行与列的关系公式，左矩阵列***n***要等于右矩阵行***n***，得到结果矩阵的行列数为A的行***m***，B的列***p***
   $$
   A_(m,n)B_(n,p) = C_(m,p)
   $$
   
2. pass



# 第二部分 AI基础

## 第一章、梯度下降算法

#### 			一、梯度下降法时一种深度学习优化算法

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

## 第二章、反向传播算法数学推导

#### 一、向前传播

1. 两层隐藏层网络向前传播计算图

![ai_bp_net_graph](C:\Users\jjc\PycharmProjects\MathFuncExercise\math_md_src\ai_bp_net_graph.png)

#### 一、单层隐藏层反向传播数学公式推导



1. w<sup>a</sup>表示是输入层与隐藏层a的权重矩阵
   $$
   f(x) = w^a*x  + b
   $$

2. 输入数据x作为隐藏层的输入，得到***h<sub>output</sub>(x)***
   $$
   y=h_{output}(x) = Sigmod(f(x))
   $$

   

3. 将隐藏层的输出 ***y*** 作为参数，计算误差值，*注：**n**为标签个数，最后一层隐藏层的神经元个数等于标签个数*
   $$
   E = \frac{1}{2}\sum_{i=1}^n(y - p_i)^2
   $$

4. 对误差***E***公式关于***w<sup>a</sup><sub>ij</sub>***进行求偏导，注：第三行***w<sup>a</sup>***矩阵与***x***矩阵相乘展开式为第四行，由于是关于***w<sup>a</sup><sub>ij</sub>***求偏导所以其他项视作常数，求导得0，即展开式求偏导得到***x<sub>j</sub>***
   $$
   \begin{aligned}
   \frac{\partial E}{\partial w^a_{ij}} &= \sum_{i=1}^n(y - p_i)\frac{\partial y}{\partial w^a_{ij}} 
   \\&= \sum_{i=1}^n(y - p_i) y (1 - y)\frac{\partial f(x)}{\partial w^a_{ij}} 
   \\&= \sum_{i=1}^n(y - p_i) y (1 - y)(w^ax+b_a)
   \\&= \sum_{i=1}^n(y - p_i) y (1 - y)(w^a_{11}x_1 + w^a_{12}x_2...+w^a_{ij}x_j + b_a)
   \\&= \sum_{i=1}^n(y - p_i) y (1 - y)x_j
   \\&= \sum_{i=1}^n(h_{output}(x_j) - p_i) h_{output}(x_j) (1 - h_{output}(x_j))x_j 
   
   \end{aligned}
   $$

5. 更新权重， η为学习率

$$
\vartriangle w^a_{ij} = -\eta\frac{\partial E}{\partial w^a_{ij}}
$$

