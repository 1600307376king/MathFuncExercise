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

## 第二章、反向传播算法数学推导

#### 多元函数表达式

1. 输入层 a代表单个神经元编号，n代表输入神经元个数
   $$
   f(x) = w*x  + b
   $$
   

2. 输入层计算输出值 fo(x)作为隐藏层的输入
   $$
   f_o(x) = sigmod(f(x))
   $$
   

3. 计算隐藏层输出
   $$
   h_o(x) = sigmod(w*f_o(x)+b)
   $$
   

4. 计算误差值
   $$
   E = \frac{1}{2}\sum_{a=1}^n(h_o(x) - p_a)^2
   $$
   

5. 对误差公式进行求偏导
   $$
   \frac{\partial E}{\partial w_i} = (h_o(x)-p_a)*(h_o(x)*(1-h_o(x))*f_o(x)*(1-f_o(x))*w_i
   $$
   

6. 更新权重

$$
\vartriangle w = -\eta\frac{\partial E}{\partial w_i}
$$

$$

$$

