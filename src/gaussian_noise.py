import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def reverse_to_img(x: torch.Tensor):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)


# 画像読み込み
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "flower.png")
image = plt.imread(file_path)
print(image.shape)

# 画像の前処理
preprocess = transforms.ToTensor()
x = preprocess(image)

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T) # noise schedule

# imgs = []

# for t in range(T):
#     if t % 100 == 0:
#         img = reverse_to_img(x)
#         imgs.append(img)

#     beta = betas[t]
#     eps = torch.randn_like(x) # noise
#     x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps # sampling from q(x_t | x_{t-1})

# plt.figure(figsize=(15, 6))
# for i, img in enumerate(imgs[:10]):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(img)
#     plt.title(f'Noise: {i * 100}')
#     plt.axis('off')

# plt.show()

def add_noise(x_0: torch.Tensor, t: int, betas: torch.Tensor):
    T = len(betas)
    assert t >= 0 and t <= T
    if t == 0:
        return x_0

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    return x_t

t_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
imgs = []

for t in t_list:
    x_t = add_noise(x, t, betas)
    img = reverse_to_img(x_t)
    imgs.append(img)

plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs):
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f'Noise: {t_list[i]}')
    plt.axis('off')
plt.show()

