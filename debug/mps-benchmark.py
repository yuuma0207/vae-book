from itertools import product

from tqdm import tqdm
import torch
import torch.utils.benchmark as benchmark


def device():
    """環境毎に利用できるアクセラレータを返す"""
    if torch.backends.mps.is_available():
        # macOS w/ Apple Silicon or AMD GPU
        return "mps"
    if torch.cuda.is_available():
        # NVIDIA GPU
        return "cuda"
    return "cpu"


def batched_dot_mul_sum(a, b):
    """mul -> sum"""
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    """bmm -> flatten"""
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


DEVICE = device()
print(f"device: {DEVICE}")


results = []

# 行列サイズ x スレッド数の組み合わせでベンチマークする
sizes = [1, 64, 1024, 10000]
for b, n in tqdm(list(product(sizes, sizes))):
    label = "Batched dot"
    sub_label = f"[{b}, {n}]"
    x = torch.ones((b, n)).to(DEVICE)
    for num_threads in [1, 4, 16, 32]:
        results.append(benchmark.Timer(
            stmt="batched_dot_mul_sum(x, x)",
            setup="from __main__ import batched_dot_mul_sum",
            globals={"x": x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="mul/sum",
        ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
            stmt="batched_dot_bmm(x, x)",
            setup="from __main__ import batched_dot_bmm",
            globals={"x": x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="bmm",
        ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.print()