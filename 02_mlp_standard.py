import torch
from torch import nn

# 我们依然沿用刚才你手写的造数据代码，生成假数据
# （假设 features 和 labels 已经存在，你可以直接拷刚才的第一步代码）
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# ==========================================
# 工业化第一步：使用 DataLoader (替代你自己写的 data_iter)
# ==========================================
from torch.utils.data import TensorDataset, DataLoader

batch_size = 10
# 将特征和标签组合成一个 PyTorch 标准数据集
dataset = TensorDataset(features, labels)
# TODO 1: 实例化一个 DataLoader。
# 要求：传入 dataset，设置 batch_size，并且要在每一轮打乱数据(shuffle=True)
...


# ==========================================
# 工业化第二步：使用 nn.Sequential 定义网络
# ==========================================
# 我们不需要再手动定义 w 和 b，也不需要手动写 X@w+b 了！
# TODO 2: 定义一个单层的全连接网络 (也叫线性层)。
# 提示: 输入特征维度是 2，输出维度是 1。查阅/使用 nn.Linear
net = nn.Sequential(
    ...
)

# 可选：初始化一下权重（PyTorch有默认初始化，但我们手动设一下更严谨）
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


# ==========================================
# 工业化第三步：定义损失函数和优化器
# ==========================================
# TODO 3: 实例化一个均方误差损失函数。 提示: 使用 nn.MSELoss()
loss = ...

# TODO 4: 实例化一个随机梯度下降优化器。
# 提示: 使用 torch.optim.SGD。你需要把 net.parameters() 传给它，并设置学习率 lr=0.03
trainer = ...


# ==========================================
# 工业化第四步：标准训练循环（重点！八股文来了！）
# ==========================================
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        # TODO 5: 前向传播，算出预测值 y_hat
        y_hat = ...
        
        # TODO 6: 计算 loss (传入预测值和真实值)
        l = ...
        
        # TODO 7: 【极其重要】把优化器里的梯度清零！(trainer...)
        ...
        
        # TODO 8: 【极其重要】反向传播，计算所有参数的梯度！(l...)
        ...
        
        # TODO 9: 【极其重要】让优化器走一步，更新参数！(trainer...)
        ...
        
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# ==========================================
# 验收结果！
# ==========================================
# 看看 nn.Linear 里面自动帮我们生成的 w 和 b 是多少
w = net[0].weight.data
b = net[0].bias.data
print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差: {true_b - b}')