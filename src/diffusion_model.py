import math
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


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
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch) # 最初はin_ch, out_chにしててバグらせた
        )
    
    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v) # multi layer perceptron
        v = v.view(N, C, 1, 1) # (N, C) -> (N, C, 1, 1) reshape
        y = self.conv(x + v)
        return y

class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x, timesteps):
        # 正弦波エンコーディング
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x
    
class Diffuser:
    def __init__(self, num_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                device="cpu"):
        
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x_0, t): 
        # x_0: torch.Tensor (N: batch size, C: channel, H: height, W: width) 
        # 複数のバッチデータを受け取るかもしれないのでall()を使う。画像の並列処理みたいなこと
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all() # 複数のバッチデータを受け取るかもしれないのでall()を使う

        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]

        N = alpha_bar.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval() # モデルを評価モードに変更
        with torch.no_grad(): # 勾配計算をしない
            eps = model(x, t) # 一つ前の学習データを使ってノイズを生成
            model.train() # モデルを学習モードに変更

            noise = torch.randn_like(x, device=self.device)
            noise[t == 1] = 0 # t=1のときはノイズを0にする
            
            mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) /\
                torch.sqrt(alpha)
            
            std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
            return mu + std * noise
    
    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)
    
    def sample(self, model, x_shape=(20, 1, 28, 28)): # Channel = 1: MNISTの白黒画像
        batch_size = x_shape[0] # 生成する画像の枚数
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t) # ワンステップのデノイズ処理を行う

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images
    

def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i], cmap="gray")
            ax.axis("off")
            i += 1
    plt.show()

    
img_size = 28
batch_size = 128 # 一回の学習で128枚の画像を使う。これを指定しないと1回の学習にすべてのデータを使う。
num_timesteps = 1000
epochs = 10 # 学習データを何回繰り返すか。1epochで全データを使う。
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root="./../data", download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(num_timesteps, device=device)
model = UNet()
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # エポックごとにデータ生成して結果を確認したい場合はコメントアウト外す
    # images = diffuser.sample(model)
    # show_images(images)

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        # 各データに対してランダムな時間ステップを選択
        # タプルとして配列の長さを渡している
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t) # 時刻tのノイズをかけたデータを生成
        noise_pred = model(x_noisy, t) # 時刻tのノイズをニューラルネットで予測
        loss = F.mse_loss(noise, noise_pred)

        loss.backward() # back propagation
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1
    print(cnt) # バッチサイズが128なので60000/128=468.75 -> 469になるはず
    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch: {epoch+1} | Loss: {loss_avg}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

images = diffuser.sample(model)
show_images(images)