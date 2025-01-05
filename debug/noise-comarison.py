import numpy as np
import matplotlib.pyplot as plt

def add_noise_for_loop(T: int, x_0: np.ndarray, betas: np.ndarray):
    x_t = x_0
    for t in range(1, T + 1):
        beta = betas[t - 1]
        eps = np.random.randn(*x_0.shape) # noise (unpacking)
        x_t = np.sqrt(1 - beta) * x_t + np.sqrt(beta) * eps # sampling from q(x_t | x_{t-1})
    return x_t

def add_noise_from_x0(T: int, x_0: np.ndarray, betas: np.ndarray):
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    alpha_bar = alpha_bars[T - 1]
    eps = np.random.randn(*x_0.shape)
    x_T = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * eps
    return x_T

def calc_cross_entropy(y_true, y_pred):
    loss = np.mean( -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), axis=0)
    return loss

init = 0

x_0 = np.array([init])
T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, T) # noise schedule

samples_1 = []
samples_2 = []
for _ in range(1000):
    sample_1 = add_noise_for_loop(T, x_0, betas)
    sample_2 = add_noise_from_x0(T, x_0, betas)
    samples_1.append(sample_1)
    samples_2.append(sample_2)

samples_1 = np.array(samples_1)
samples_2 = np.array(samples_2)

fig = plt.figure()
plt.hist(samples_1, bins=50, color='blue', alpha=0.5)
plt.hist(samples_2, bins=50, color='red', alpha=0.5)

plt.show() # 同一の分布になっていることを確認