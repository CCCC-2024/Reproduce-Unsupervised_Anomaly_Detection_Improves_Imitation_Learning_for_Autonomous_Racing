***\*核心原因其实是：\*\*论文并没有“选 best checkpoint”这一套定义\*\*，而是\**固定训练轮数**，然后用固定轮数下的模型去画 PCC 曲线/做检测结果。

### 1) 论文设定本来就是“固定 epoch 的 checkpoint”

论文明确写了：CAE 在 **raindrop & plastic 训练 100 epochs**，在 **wall-hit 训练 10 epochs**。Unsupervised_Anomaly_Detection_…

而且 Fig.4 的检测结果也是按这个 epoch 数汇报的（raindrop/plastic=100，wall-hit=10）。Unsupervised_Anomaly_Detection_…

### 2) 这篇工作“没有验证集”，所以严格来说也没法定义 best.pt

论文还强调：他们**不留 validation set 来防止 overfitting**，因为这个设定就是要“在混合数据上过拟合 normal，同时不要过拟合 abnormal”。Unsupervised_Anomaly_Detection_…

所以如果你问“best”是按什么选：

- 按 val loss？论文没有 val。
- 按 detection 指标？论文是“完全无监督”，也不靠标签调参（你现在是为了 sanity check 才有 y_dirty）。

因此**复现导向最稳妥的默认就是：用论文指定 epoch 的最后一个 checkpoint**，也就是 `ckpt_ep100.pt`（或 wall-hit 的 `ckpt_ep010.pt`）。

### 3) 那我们 repo 里为什么没有 best.pt？

因为我们现在的保存策略是“按 epoch 存档”：`ckpt_ep001.pt / ckpt_ep002.pt / ...`（你现在 debug_small 就已经产生了 `ckpt_ep002.pt`）。
 **best.pt 需要你先定义 best 的准则**，否则写一个 best 反而会让复现目标变得不清晰。

### 4) 你现在该怎么用（结合你本地现状）

你当前跑的是 debug_small（2 epochs），所以 `run_score_pcc.py` 里应当指向：

- `CKPT_PATH = "outputs/cae/debug_small/ckpt_ep002.pt"`

而我之前写 `ckpt_ep100.pt`，是对应“把 epochs 设为 100、跑完整 raindrop/plastic 论文设定”之后的路径。

### 5) 如果你**强烈想要 best.pt**，也可以，但要先选一种“best 定义”

给你三个**可复现、可解释**的方案（你选哪种都行）：

1. **best = 最小训练 loss 的 epoch**（最简单，但不一定等价于最好检测）
2. **best = 最大 clean/dirty 的 PCC 分离度**（比如 median(PCC_clean) − median(PCC_dirty) 最大；更贴近检测目标）
3. **best = 用你现在的 y_dirty 做一个“仅用于 debug 的选择”**（比如 F1/precision/recall 最好；但要在 README 里声明：这是 debug 方便，不是论文无监督设定）