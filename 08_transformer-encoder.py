import torch
from torch import nn

# 假设你的 MultiHeadAttention 已经写好了，我们这里为了演示直接用官方的
# (在实际中，你可以把你上一关写的类 import 进来！)
# 这里我们直接使用 PyTorch 自带的 nn.MultiheadAttention 替代

class PositionWiseFFN(nn.Module):
    """小零件：基于位置的前馈网络 (其实就是个两层的 MLP)"""
    def __init__(self, d_model, ffn_hidden):
        super().__init__()
        # TODO 1: 定义两个线性层和一个激活函数
        # 第一层: 输入 d_model，输出 ffn_hidden
        # 激活函数: ReLU (现在大模型更流行用 GELU，这里用 ReLU 即可)
        # 第二层: 输入 ffn_hidden，输出 d_model
        self.linear1 = ...
        self.relu = ...
        self.linear2 = ...

    def forward(self, X):
        # TODO 2: 按顺序前向传播
        return ...


class TransformerEncoderBlock(nn.Module):
    """绝对基石：Transformer 编码器块"""
    def __init__(self, d_model, num_heads, ffn_hidden, dropout=0.1):
        super().__init__()
        
        # 核心组件 1：多头注意力机制
        # 注意: PyTorch官方的 MHA 默认输入形状是[seq_len, batch, d_model]
        # 如果要用 [batch, seq_len, d_model]，需要加 batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        
        # 核心组件 2：前馈神经网络 (调用上面你写的)
        self.ffn = PositionWiseFFN(d_model, ffn_hidden)
        
        # 核心组件 3：两个层归一化 (LayerNorm)
        # LayerNorm 是大模型稳定训练的灵魂，它对最后一个维度(d_model)求均值和方差
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X):
        # 假设 X 的形状: [batch_size, seq_len, d_model]
        
        # ==========================================
        # 第一步：Multi-Head Attention + Add & Norm
        # ==========================================
        # 1. 算出注意力结果 (Q, K, V 都是 X)
        # 官方的 attention 返回两个值: (输出结果, 注意力权重)，我们只需要第一个，所以用 [0]
        attn_out = self.attention(X, X, X)[0]
        
        # TODO 3: 实现第一个 Add & Norm (极其重要的大模型八股文！)
        # 将 attn_out 过一下 dropout1，然后和原始的输入 X 相加 (残差连接)
        # 最后把相加的结果扔进 norm1 中。并将结果覆盖到 X 上。
        X = self.norm1(X + ...)
        
        # ==========================================
        # 第二步：Feed Forward Network + Add & Norm
        # ==========================================
        # 2. 把更新后的 X 扔进 FFN 中
        ffn_out = self.ffn(X)
        
        # TODO 4: 实现第二个 Add & Norm
        # 将 ffn_out 过一下 dropout2，然后和当前的 X 相加
        # 最后把相加的结果扔进 norm2 中。
        X = ...
        
        return X

# ==========================================
# 终极测试：验证形状匹配 (SSD)
# ==========================================
if __name__ == "__main__":
    d_model = 512       # 词向量维度
    num_heads = 8       # 8个头
    ffn_hidden = 2048   # FFN 内部放大到 2048 维
    batch_size = 2
    seq_len = 50        # 一句话 50 个词
    
    # 实例化一个 Encoder 块
    encoder_block = TransformerEncoderBlock(d_model, num_heads, ffn_hidden)
    
    # 造一个假句子输入 [2, 50, 512]
    X = torch.rand(batch_size, seq_len, d_model)
    
    # 扔进块里算一下
    out = encoder_block(X)
    
    print(f"输入 X 形状: {X.shape}")
    print(f"输出 形状: {out.shape}") # 期待完美输出[2, 50, 512]