import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sympy
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def sigmoid(sig_val):
    return 1 / (1 + np.exp(-1 * sig_val))


class BPNet(object):
    """
    网络为3层网络，输入层有三个神经元， 隐藏层4个，输出层1个
    test_x 输入层数据
    layer_number 总层数
    w0 为输入层到隐藏层的初始参数
    b0 为输入层到隐藏层的初始偏执
    node_list 为每层节点个数列表
    """

    def __init__(self, test_x):
        self.test_x = test_x
        self.layer_number = 3
        row, col = np.shape(test_x)
        self.w0 = None
        self.b0 = 0
        self.node_list = [4, 4, 1]

    @staticmethod
    def default_init_weight_fun(m, n):
        """

        :param m: 第i-1层输入节点数
        :param n: 第i层输出节点数
        :return:
        """
        min_val = -4 * np.sqrt(6) / np.sqrt(m + n)
        max_val = 4 * np.sqrt(6) / np.sqrt(m + n)
        return np.mat(np.random.uniform(min_val, max_val, size=(m, n)))

    def forward(self):
        """
        向前传播
        temp_val 每层的x值
        :return:
        """
        temp_val = self.test_x
        for i in range(1, self.layer_number):
            temp_w0 = self.default_init_weight_fun(self.node_list[i], temp_val.shape[0])
            temp_val = sigmoid(temp_w0 * temp_val + self.b0)
        return temp_val


class GradientDescent(object):
    def __init__(self, alpha=0.01, epochs=1000):
        """
        梯度下降
        :param alpha: 步长
        :param epochs: 迭代次数
        """
        self.alpha = alpha
        self.epochs = epochs

    def compute_val_for_lr(self, init_weight_vector, partial_val_function):
        epoch = 0
        loss_val_list = []

        while epoch < self.epochs:
            partial_val, loss_val = partial_val_function(init_weight_vector, epoch, True)
            init_weight_vector = init_weight_vector - self.alpha * partial_val
            loss_val_list.append(loss_val)
            epoch += 1

        return init_weight_vector, loss_val_list, self.epochs

    def compute_val_for_soft(self, init_weight_matrix, partial_val_function):
        epoch = 0
        loss_val_list = []
        while epoch < self.epochs:
            partial_val, loss_val = partial_val_function(init_weight_matrix, epoch, True)
            init_weight_matrix = init_weight_matrix - self.alpha * partial_val
            loss_val_list.append(loss_val)
            epoch += 1
        return init_weight_matrix, loss_val_list, self.epochs


class LogisticRegression(object):
    def __init__(self, data, data_label):
        """
        data 数据矩阵 numpy matrix
        m 样本数量
        n 样本特征数
        y 数据标签
        w 数据权重
        b 偏置

        设 w0 = b, x0 = 1
        f(x) = wx + b -> wx + w0x0 -> data * weight
        :param data:
        """
        self.data = data
        self.m, self.n = data.shape
        self.y = data_label
        self.w = np.random.uniform(0, 1, size=(self.n, 1))
        self.b = 0
        self.loss_val_list = []
        self.epochs = 0

    def loss_func_partial(self, new_w, iterations, is_print_loss=False):
        """
        损失函数关于第j个权重分量的偏导
        :param iterations: 迭代次数
        :param is_print_loss: 是否打印计算过程中的损失函数值
        :param new_w: 更新的权重值
        :return: 偏导值
        """
        sigmoid_val = sigmoid(self.data * new_w)
        loss_val = 0
        if is_print_loss:
            loss_val = self.get_loss_val(sigmoid_val, self.y)
            print('-------iterations = ' + str(iterations) + 'loss_val = ' + str(loss_val))
        partial_val = -1 / self.m * self.data.T * (self.y - sigmoid_val)
        return partial_val, loss_val

    def get_loss_val(self, sigmoid_val, label):
        """
        计算损失函数值
        :param sigmoid_val: 激活函数值
        :param label: 数据标签
        :return: 函数损失值
        """
        loss = 0
        for i in range(self.m):
            loss = - (label[i, 0] * np.log(sigmoid_val[i, 0]) + (1 - label[i, 0]) * np.log(1 - sigmoid_val[i, 0]))
        return loss / self.m

    def update_weight(self):
        """
        通过梯度下降更新权重
        :return: 最新权重值
        """
        init_weight_vector = self.w
        gd = GradientDescent(alpha=0.1)
        new_weight_vector, self.loss_val_list, self.epochs = \
            gd.compute_val_for_lr(init_weight_vector, self.loss_func_partial)
        return new_weight_vector

    def display_loss_plot(self):
        """
        显示损失函数变化图
        :return:
        """
        if self.loss_val_list and self.epochs == len(self.loss_val_list):
            plt.title('损失函数变化图')
            plt.xlabel('迭代次数')
            plt.ylabel('损失值')
            plt.plot(range(self.epochs), self.loss_val_list)
            plt.show()


class SoftMaxRegression(object):
    def __init__(self, data, label):
        """
        m: 数据行数
        k: 类别数
        w: 初始权重（mxk矩阵）
        y: 标签（kx1矩阵）
        :param data: 输入数据
        :param label: 标签类别
        """
        self.m, self.n = np.shape(data)
        self.data = data
        self.y = label
        self.k = len(set(self.y.flatten().tolist()[0]))
        self.w = np.mat(np.ones((self.n, self.k)))
        self.loss_val_list = []
        self.epochs = 0

    def indicator_func(self, label_matrix):
        """
        分类矩阵，行：样本，列：类别
        [[0, 1],
         [1, 0],
         [1, 0],
            ]
        :param label_matrix:
        :return:
        """
        init_label = np.mat(np.zeros((self.m, self.k)))
        for j in range(self.m):
            init_label[j, label_matrix[j, 0]] = 1
        return init_label

    def get_loss_func(self, new_weight):
        """
        获取损失值
        :param new_weight:
        :return:
        """
        loss = 0
        for i in range(self.m):
            e_theta_w = np.sum(self.data[i] * new_weight[:, self.y[i, 0]])
            all_e_theta_w = np.sum([self.data[i] * new_weight[:, l] for l in range(self.k)])
            loss += -1 / self.m * np.log(e_theta_w / all_e_theta_w)
        return loss

    def loss_func_partial(self, new_weight, iterations, is_print_loss=False):
        """
        损失函数怕偏导值
        :param new_weight:
        :param iterations:
        :param is_print_loss:
        :return:
        """
        loss_val = self.get_loss_func(new_weight)
        if is_print_loss:
            if iterations % 100 == 0:
                print("------iterations = " + str(iterations) + ',loss_val = ' + str(loss_val))
        new_weight_vector = -1 / self.m * self.data.T * \
                            (self.indicator_func(self.y) - np.exp(self.data * new_weight) / np.exp(
                                self.data * new_weight).sum(axis=1).repeat(self.k, axis=1))
        return new_weight_vector, loss_val

    def update_weight(self):
        """
        更新权重
        :return:
        """
        init_weight_matrix = self.w
        gd = GradientDescent()
        new_weight_matrix, self.loss_val_list, self.epochs = \
            gd.compute_val_for_soft(init_weight_matrix, self.loss_func_partial)
        return new_weight_matrix

    def display_loss_plot(self):
        """
        SoftMax显示损失函数变化图
        :return:
        """
        if self.loss_val_list and self.epochs == len(self.loss_val_list):
            plt.title('损失函数变化图')
            plt.xlabel('迭代次数')
            plt.ylabel('损失值')
            plt.plot(range(self.epochs), self.loss_val_list)
            plt.show()

    @staticmethod
    def test_data_predict(last_weight, test_data, test_label):
        test_data_length = len(test_data)
        err_count_number = 0
        for row in range(test_data_length):
            if test_label[row] - (test_data * last_weight).argmax(axis=1)[row] > 0:
                err_count_number += 1
        accuracy = 1 - (err_count_number / test_data_length)
        print("test data predict accuracy is {0} %".format(accuracy * 100))


iris = datasets.load_iris()

# test_data = np.mat([[1, 2, 1],
#                     [3, 3, 1],
#                     [1, 4, 1],
#                     [2, 4, 1]])
#
# test_data_label = np.mat([1, 1, 0, 0]).T

train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# lr = LogisticRegression(test_data, test_data_label)
# print(lr.update_weight())
# lr.display_loss_plot()

sm = SoftMaxRegression(train_x, np.mat([train_y]).T)
new_w = sm.update_weight()
print(new_w)
sm.display_loss_plot()
sm.test_data_predict(new_w, test_x, np.mat([test_y]).T)

# print(sm.update_weight())
# epoch = 0
# init_vec = lr.w
# loss_val_list = []
# while epoch < 1000:
#     partial_val, loss_val = lr.loss_func_partial(init_vec, epoch, True)
#     init_vec = init_vec - 0.1 * partial_val
#     loss_val_list.append(loss_val)
#     epoch += 1
#
# print(init_vec)
# plt.plot(range(1000), loss_val_list)
# plt.show()
# a = np.array(range(len(loss_list)))
# plt.scatter(a, [i[0] for i in loss_list])
# plt.scatter(a, [i[1] for i in loss_list])
# plt.show()
# init_b = lr.b
# epoch = 0
# while epoch < 1000:
#
#     print(init_b)
#     init_b = init_b - 0.01 * np.array([[lr.bias_loss_fun()]])
#     epoch += 1
#     if init_b[0] < 1e-3 or init_b[1] < 1e-3:
#         break
#     epoch += 1
