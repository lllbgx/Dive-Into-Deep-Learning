import torch
from torch import nn

class HandTornRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # ==========================================
        # 核心权重定义
        # RNN 公式: H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b)
        # ==========================================
        # TODO 1: 定义输入 x 到隐藏状态 h 的线性层
        # 提示: 使用 nn.Linear，输入维度是 input_size，输出维度是 hidden_size
        self.W_xh = ...
        
        # TODO 2: 定义上一时刻 h_{t-1} 到当前 h_t 的线性层
        # 提示: 使用 nn.Linear，输入和输出都是 hidden_size。
        # 注意: 因为上面的 W_xh 已经自带了偏置项(bias)，这里可以设 bias=False 避免重复加偏置
        self.W_hh = ...
        
        # 激活函数
        self.tanh = nn.Tanh()

    def forward(self, X, H=None):
        # 假设 X 的形状是: [batch_size, seq_len, input_size]
        # (比如：32句话，每句话10个词，每个词被表示为长512的向量)
        batch_size, seq_len, _ = X.shape
        
        # ==========================================
        # 初始化“记忆”
        # ==========================================
        # TODO 3: 如果没有传入初始的记忆 H，我们需要初始化一个全 0 的 H
        # 极其关键: 记忆 H 并不关心句子有多长(seq_len)，它只关心 batch 里的每一个样本！
        # 提示: H 的形状必须是 (batch_size, self.hidden_size)
        if H is None:
            H = torch.zeros((..., ...), device=X.device)
            
        # 用一个列表来保存网络在每一个时间步(每一个词)上的输出
        outputs =[]
        
        # ==========================================
        # 终极奥义：时间步循环 (Unrolling in Time)
        # ==========================================
        # TODO 4: 写一个 for 循环，遍历所有的序列长度(seq_len)
        for t in range(...):
            # 取出当前时刻的输入 X_t
            # 提示: X 的形状是[batch, seq, feature]，我们要切片取出第 t 个时间步
            X_t = X[:, t, :]  # 切片后 X_t 的形状是 [batch_size, input_size]
            
            # TODO 5: RNN 的绝对核心公式！计算当前时刻的新 H
            # 把 X_t 扔进 self.W_xh()，把旧的 H 扔进 self.W_hh()，两者相加，最后套上 self.tanh()
            # 注意: 这个新的 H 会在下一次循环时，变成“旧的 H”！
            H = ...
            
            # 把当前时刻产生的记忆保存到列表里
            outputs.append(H)
            
        # 把列表里的所有 Tensor 沿着序列长度维度(dim=1)拼接起来
        # 形状变化: list of [batch, hidden] -> [batch, seq_len, hidden]
        out = torch.stack(outputs, dim=1)
        
        return out, H

# ==========================================
# 终极测试：验证形状匹配 (SSD)
# ==========================================
if __name__ == "__main__":
    # 模拟数据设定
    batch_size = 4      # 同时处理 4 句话
    seq_len = 7         # 每句话有 7 个词 (7个时间步)
    input_size = 128    # 每个词被编码成长度为 128 的向量
    hidden_size = 256   # 我们希望 RNN 维持一个长度为 256 的记忆向量
    
    # 1. 实例化我们手撕的 RNN
    rnn = HandTornRNN(input_size, hidden_size)
    
    # 2. 造一个假的三维序列输入
    X = torch.rand(batch_size, seq_len, input_size)
    
    # 3. 运行前向传播
    out, final_H = rnn(X)
    
    # 如果没报错，来看看形状是不是对的！
    print(f"输入 X 形状: {X.shape}")
    print(f"全部输出 out 形状: {out.shape}")      # 期待:[4, 7, 256]
    print(f"最终记忆 final_H 形状: {final_H.shape}") # 期待: [4, 256]