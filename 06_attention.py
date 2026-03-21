import math
import torch
from torch import nn
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # 假设 queries 的形状: [B, num_queries, d]
        # 假设 keys 的形状:[B, num_kv_pairs, d]
        # 假设 values 的形状:[B, num_kv_pairs, v_dim]
        
        # TODO 1: 提取特征维度 d 的大小
        # 提示: d 就是 queries 的最后一个维度的大小 (使用 .shape[-1])
        d = ...
        
        # TODO 2: 计算 Q 和 K^T 的点积 (即 Q 乘 K的转置)
        # 极其关键: K 目前的形状是 [B, num_kv_pairs, d]
        # 我们需要把它变成[B, d, num_kv_pairs]，也就是把最后两个维度交换！
        # 提示: 使用 keys.transpose(1, 2) 或 keys.transpose(-2, -1)
        # 然后使用 @ 运算符进行矩阵乘法: queries @ K的转置
        # 预期 scores 的形状:[B, num_queries, num_kv_pairs]
        scores = ...
        
        # TODO 3: 缩放操作 (除以 d 的平方根)
        # 提示: math.sqrt(d) 返回的是一个浮点数，直接除就行
        scores = ...
        
        # TODO 4: 在最后一个维度上做 Softmax 操作，得到注意力权重
        # 提示: 使用 F.softmax(..., dim=-1)
        attention_weights = ...
        
        # 为了防止过拟合，对权重做一下 dropout (可以直接调 self.dropout)
        attention_weights = self.dropout(attention_weights)
        
        # TODO 5: 将注意力权重与 V 相乘
        # 注意力权重的形状是 [B, num_queries, num_kv_pairs]
        # V 的形状是 [B, num_kv_pairs, v_dim]
        # 提示: 再次使用 @ 运算符相乘
        # 最终输出 out 的形状应该是[B, num_queries, v_dim]
        out = ...
        
        return out

# ==========================================
# 终极测试：验证形状匹配 (SSD)
# ==========================================
if __name__ == "__main__":
    # 我们来造一些假数据 (Batch_size=2)
    # 假设有 2 个查询，每个查询维度是 4
    queries = torch.normal(0, 1, (2, 2, 4))
    
    # 假设有 10 个键值对，K 的维度必须和 Q 一样是 4
    keys = torch.normal(0, 1, (2, 10, 4))
    
    # 假设 V 的维度是 8
    values = torch.normal(0, 1, (2, 10, 8))
    
    attention = DotProductAttention(dropout=0.5)
    
    # 开启测试模式(关闭dropout)以便测试
    attention.eval() 
    out = attention(queries, keys, values)
    
    print(f"输入 Queries 形状: {queries.shape}")
    print(f"输出 Out 形状: {out.shape}") # 必须完美输出 [2, 2, 8]