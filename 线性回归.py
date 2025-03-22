import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('house_data.csv')
print(df.isnull().sum())
def manual_standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized = (data - mean) / std
    return standardized, mean, std
X = df.drop('MEDV', axis=1).values
y = df['MEDV'].values

# 标准化特征
X_std, X_mean, X_std_dev = manual_standardize(X)
print("标准化后的特征示例：\n", X_std[:2])


# 分割训练集和测试集
def train_test_split(X, y, test_size=0.2):
    np.random.seed(50)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    X_train = X[indices[:split]]
    X_test = X[indices[split:]]
    y_train = y[indices[:split]]
    y_test = y[indices[split:]]
    return X_test, X_train, y_test, y_train

X_train, X_test, y_train, y_test = train_test_split(X_std, y)

print('训练集大小：', X_train.shape[0], "测试集大小：", X_test.shape[0])

# 数据可视化
plt.figure(figsize=(8, 4))
plt.hist(y, bins=25, edgecolor='k')
plt.xlabel('MEDV')
plt.ylabel('i')

plt.figure(figsize=(8, 4))
plt.scatter(X[:, 5], y, alpha=0.5)
plt.title('RM vs MEDV')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []  # 添加 loss_history 属性

    def fit(self, X, y):
        y = y.reshape(-1, 1)  # 将 y 转换为列向量
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏置项
        self.theta = np.zeros((X_b.shape[1], 1))  # 初始化参数为列向量
        m = len(y)

        for _ in range(self.epochs):
            gradients = (2 / m) * X_b.T @ (X_b @ self.theta - y)  # 梯度公式
            self.theta -= self.learning_rate * gradients  # 更新参数
            loss = np.mean((X_b @ self.theta - y) ** 2)  # 计算损失
            self.loss_history.append(loss)  # 记录损失

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta
# 训练模型
model_gd = LinearRegressionGradientDescent(learning_rate=0.01, epochs=1000)
model_gd.fit(X_train, y_train)
y_pred_gd = model_gd.predict(X_test)


# 输出均方误差
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


mse_gd = mean_squared_error(y_test, y_pred_gd)
print("梯度下降MSE：", mse_gd)

# 梯度下降曲线
plt.figure(figsize=(8, 4))
plt.plot(range(model_gd.epochs), model_gd.loss_history, color='red')
plt.title('梯度下降损失曲线')
plt.xlabel('n')
plt.ylabel('MSE')
plt.grid(True)
plt.show()