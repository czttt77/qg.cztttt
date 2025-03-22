import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = df[df["class"].isin(["Iris-setosa", "Iris-versicolor"])]

# 提取特征和标签（使用花瓣长度和宽度）
X = df[["petal_length", "petal_width"]].values
y = (df["class"] == "Iris-versicolor").astype(int).values  # 变色鸢尾为1，山鸢尾为0

# 标准化特征
def manual_standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized = (data - mean) / std
    return standardized, mean, std

X_std, X_mean, X_std_dev = manual_standardize(X)

# 分割数据集
def train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X_std, y)

# 逻辑回归模型实现
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []  # 记录损失值变化

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X_b.shape[1])
        m = len(y)

        for epoch in range(self.epochs):
            # 计算预测值和损失
            z = X_b @ self.theta
            h = self.sigmoid(z)
            loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
            self.loss_history.append(loss)
            # 计算梯度并更新参数
            gradient = (1/m) * X_b.T @ (h - y)
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)

# 训练模型
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("测试集准确率：", accuracy)

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(range(model.epochs), model.loss_history, color='red')
plt.title('Logistic Regression Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

# 绘制测试集数据点
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
            color='blue', label='Iris-setosa', edgecolor='k', alpha=0.7)
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
            color='red', label='Iris-versicolor', edgecolor='k', alpha=0.7)

# 生成网格点用于绘制决策边界
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 预测网格点类别
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.title('Iris Classification with Decision Boundary')
plt.legend()
plt.show()