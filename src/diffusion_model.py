import torch
from torch import nn


# 正弦波エンコーディング
# 整数tに対して、出力次元output_dimのテンソルを返す。
# iが偶数のとき sin(t / 10000^(i/output_dim))
# iが奇数のとき cos(t / 10000^(i/output_dim))
def _pos_encoding(t, output_dim, device="cpu"):
    """
    t: time
    output_dim: output dimension
    device: cpu or cuda

    return: tensor of shape (output_dim,)
    """
    D = output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = 10000 ** (i / D)

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


# バッチデータ処理用の関数
def pos_encoding(ts, output_dim, device="cpu"):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
