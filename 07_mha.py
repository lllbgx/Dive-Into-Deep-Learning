import torch
from torch import nn

# 把你刚才写的 DotProductAttention 拷过来（这里我用简写代替）
class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / (d ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(attention_weights, values)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model: 输入的特征维度 (例如 512)
        num_heads: 头的数量 (例如 8)
        注意: d_model 必须能被 num_heads 整除 (512 / 8 = 64，每个头的维度是 64)
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # 定义生成 Q, K, V 的三个线性映射层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 最后的输出线性层
        self.W_o = nn.Linear(d_model, d_model)
        
        # 实例化我们上一关写好的点积注意力
        self.attention = DotProductAttention()

    def split_heads(self, X):
        """极其关键的拆头函数！"""
        # 假设输入 X 的形状是: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = X.shape
        head_dim = d_model // self.num_heads
        
        # TODO 1: 施展你的空间魔法，完成拆头三步曲！
        # 第 1 步: 用 view 将 X 的形状变成[batch_size, seq_len, num_heads, head_dim]
        # 第 2 步: 用 transpose 或 permute 交换 seq_len 和 num_heads 的位置
        #          变成[batch_size, num_heads, seq_len, head_dim]
        # 第 3 步: 用 view 强行合并前两个维度！(相当于把多头和batch当成同一个维度一起丢进注意力去算)
        #          变成[batch_size * num_heads, seq_len, head_dim]
        # 提示: 如果 view 报错说张量不连续(not contiguous)，就在前面加个 .contiguous()，即 X.contiguous().view(...)
        
        X = ...
        return X

    def combine_heads(self, X, batch_size):
        """极其关键的拼头函数！"""
        # 输入 X 是注意力算完的结果，形状: [batch_size * num_heads, seq_len, head_dim]
        # 我们要把它变回 [batch_size, seq_len, d_model]
        
        # TODO 2: 逆向施展魔法！
        # 第 1 步: 用 view 变成[batch_size, self.num_heads, seq_len, head_dim]
        # 第 2 步: 交换 num_heads 和 seq_len 回来，变成[batch_size, seq_len, self.num_heads, head_dim]
        # 第 3 步: 再次用 view 把最后两个维度合并，变成 [batch_size, seq_len, self.d_model]
        
        X = ...
        return X

    def forward(self, queries, keys, values):
        batch_size = queries.shape[0]
        
        # 1. 经过线性层
        Q = self.W_q(queries)
        K = self.W_k(keys)
        V = self.W_v(values)
        
        # 2. 拆分多头 (调用你写的拆头魔法)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. 扔进点积注意力引擎
        # 此时传进去的形状全是 [batch*heads, seq_len, head_dim]
        attn_out = self.attention(Q, K, V)
        
        # 4. 把多头的结果重新拼回去
        out = self.combine_heads(attn_out, batch_size)
        
        # 5. 最后过一个线性层收尾
        return self.W_o(out)

# ==========================================
# 终极测试：验证形状匹配 (SSD)
# ==========================================
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    # 造假数据 [2, 10, 512]
    X = torch.normal(0, 1, (batch_size, seq_len, d_model))
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # 模拟自注意力机制 (Q, K, V 全是同一个 X)
    out = mha(X, X, X)
    
    print(f"输入 X 形状: {X.shape}")
    print(f"多头注意力输出 形状: {out.shape}") # 期待完美输出[2, 10, 512]