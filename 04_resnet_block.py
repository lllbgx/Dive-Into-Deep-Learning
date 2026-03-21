import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        
        # TODO 1: 定义第一个卷积层
        # 要求：输入 input_channels，输出 num_channels，卷积核大小为 3，填充 padding=1
        # 极其重要：步幅 stride 必须设为传入的 strides！(这决定了是否降采样)
        self.conv1 = ...
        
        # TODO 2: 定义第二个卷积层
        # 要求：输入和输出都是 num_channels！卷积核大小为 3，填充 padding=1。
        # 注意：这里的步幅 stride 必须是 1（不再降采样了）
        self.conv2 = ...
        
        # TODO 3: 定义一个 1x1 卷积层 (如果 use_1x1conv 为 True 的话)
        # 这个层的作用是用来调整 X 的形状，使其能和 F(X) 相加！
        if use_1x1conv:
            # 要求：输入 input_channels，输出 num_channels，卷积核大小为 1
            # 步幅必须设为传入的 strides！
            self.conv3 = ...
        else:
            self.conv3 = None
            
        # 批量归一化层 (不用改)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, X):
        # ==========================================
        # 前向传播：计算 F(X)
        # ==========================================
        # TODO 4: 把 X 依次通过: conv1 -> bn1 -> relu -> conv2 -> bn2
        # 注意！这里存到一个新变量 Y 里面，不要覆盖掉原来的 X！
        Y = ...
        
        # ==========================================
        # 残差连接核心逻辑：Y = F(X) + X
        # ==========================================
        # TODO 5: 判断是否使用了 1x1 卷积
        if self.conv3:
            # 如果用了，把 X 扔进 conv3 变一下形状
            X = ...
            
        # TODO 6: 把 X 加上 Y，再通过一个 ReLU 激活函数，最后返回
        Y += X
        return self.relu(Y)

# ==========================================
# 终极测试：验证形状匹配 (SSD)
# ==========================================
if __name__ == "__main__":
    # 测试 1：输入和输出形状完全一致的普通残差块
    blk1 = Residual(input_channels=3, num_channels=3)
    X = torch.rand(4, 3, 224, 224) # 模拟 4 张 224x224 的 RGB 图片
    Y1 = blk1(X)
    print(f"测试1 - 预期形状: [4, 3, 224, 224], 实际形状: {Y1.shape}")

    # 测试 2：高宽减半，且通道数翻倍的残差块 (降采样块)
    blk2 = Residual(input_channels=3, num_channels=6, use_1x1conv=True, strides=2)
    Y2 = blk2(X)
    # 因为 strides=2，高宽 224/2=112；因为 num_channels=6，通道变成6
    print(f"测试2 - 预期形状:[4, 6, 112, 112], 实际形状: {Y2.shape}")