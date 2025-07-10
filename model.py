import torch
import torch.nn as nn
# import dgl
# from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from dlutils import *
# from src.constants import *
torch.manual_seed(1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# class TranAD_Basic(nn.Module):
#     """
#     基础版TranAD模型，基于Transformer的编码器-解码器架构，实现多变量时序数据的异常检测。
#     对应《TranAD.pdf》3.3节“Transformer Model”的基础结构，未包含论文中的自条件化和对抗训练模块。
#     """
#     def __init__(self, feats):
#         super(TranAD_Basic, self).__init__()
#         self.name = 'TranAD_Basic'  # 模型名称
#         self.lr = lr  # 学习率（需从外部传入，论文中未明确具体值，需根据实验调整）
#         self.batch = 128  # 批次大小（用于训练时的批量处理）
#         self.n_feats = feats  # 输入特征维度（多变量时序的变量数，如论文实验中的m=28）
#         self.n_window = 10  # 滑动窗口大小K，对应论文中的local contextual window（公式(1)中的Wt）
#         self.n = self.n_feats * self.n_window  # 窗口特征总维度（n_feats * 窗口长度）
#
#         # 位置编码层（对应论文3.3节）
#         self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
#
#         # 编码器
#         # 输入维度d_model=feats，与论文中“多变量输入”一致，nhead=feats为多头注意力头数
#         encoder_layers = TransformerEncoderLayer(
#             d_model=feats,
#             nhead=feats,
#             dim_feedforward=16,  # 前馈网络维度，论文未明确，经验值设为16
#             dropout=0.1  # 丢弃率，防止过拟合
#         )
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)  # 单层编码器
#
#         # 解码器配置（与编码器对称）
#         decoder_layers = TransformerDecoderLayer(
#             d_model=feats,
#             nhead=feats,
#             dim_feedforward=16,
#             dropout=0.1
#         )
#         self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=1)  # 单层解码器
#
#         self.fcn = nn.Sigmoid()  # 最终激活函数，输出范围[0,1]，匹配归一化数据
#
#     def forward(self, src, tgt):
# 		"""
#         前向传播逻辑，实现Transformer的编码-解码流程。
#         参数：
#         src: 输入序列，形状为 [seq_len, batch_size, n_feats]
#         tgt: 形状为 [seq_len, batch_size, n_feats]
#         返回：
#         x: 重构后的序列，形状为 [seq_len, batch_size, n_feats]
#         """
# 		# 1. 输入缩放（Scaling）
# 		src = src * math.sqrt(self.n_feats)
#         # 作用：对输入特征进行缩放，稳定训练过程中的梯度传播
#         # 理论依据：论文3.3节提到的“scaled-dot product attention”机制，通过缩放防止注意力权重过大
#
#         # 2. 位置编码（Positional Encoding）
#         src = self.pos_encoder(src)
#         # ！！！！！这里是重点，位置编码的是w,从此处往后面的src都是w
#         # 3. Transformer编码器
#         memory = self.transformer_encoder(src)
#         # 作用：通过多头注意力和前馈网络提取全局时序特征
#         # memory形状：[seq_len, batch_size, d_model]（d_model=feats）
#         # 对应论文：编码器输出对应图1中的“Encoder”模块，生成上下文向量I1^2
#
#         # 4. Transformer解码器
#         x = self.transformer_decoder(tgt, memory)
#         # 作用：根据编码器输出（memory）重构目标序列tgt
#         # 输入参数：
#         #   tgt: 目标序列（需与src同形）
#         #   memory: 编码器生成的上下文向量
#         # 对应论文：解码器对应图1中的“Decoder”模块，生成重构序列O1
#
#         # 5. 激活函数
#         x = self.fcn(x)
#         # 作用：通过Sigmoid将输出归一化至[0,1]，匹配输入数据的归一化范围
#         # 理论依据：论文3.4节提到的“重构误差计算”前提是输入输出范围一致
#
#         return x