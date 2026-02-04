# TEA-LAB — 论文复现：Unsupervised Anomaly Detection Improves Imitation Learning for Autonomous Racing

本仓库以“可复现”为第一目标，复现论文 **Unsupervised Anomaly Detection Improves Imitation Learning for Autonomous Racing** 的核心流程：  
使用 **卷积自编码器（CAE）+ latent reference loss** 学习正常示例的重构特征，并基于 **PCC（Pearson Correlation Coefficient）** 对重构质量打分，通过 **median filter + 阈值** 检测异常帧，实现示例数据清洗；后续再将清洗后的数据用于 **Behavior Cloning（BC）** 训练，比较 all-data vs cleaned-data 的性能差异。

> 当前仓库包含的 `16000clean7dirty.pkl` 为 **图像数据**（clean + 7类dirty）。  
> BC/控制器训练阶段需要 steering/actions 标签数据，拿到后即可接入并完成论文的“imitation learning improvement”部分。

---

## 0) 复现目标（验收点）

### A. 异常检测 + 清洗（优先完成，对齐 Table II 思路）
对每个异常类型（例如 `raindrop / plastic / hitwall`）完成：
1. 构造混合数据（clean : dirty = 10 : 1）
2. 训练 CAE（含 latent reference loss）
3. 计算每帧 PCC 分数（输入 vs 重构）
4. median filter 平滑（窗口约 100）
5. 阈值规则：`delta = median(PCC_smooth) - 0.05`，若 `PCC_smooth < delta` 判异常
6. 用混合数据的 GT dirty 标签评估 **Precision / Recall（Table II 风格）**
7. 导出 clean mask 与 cleaned 数据包，供后续 BC 使用

### B. 模仿学习提升（后续）
拿到 steering/actions 标签后：
- 训练 BC 控制器
- 对比 all-data vs cleaned-data 的指标（论文使用 CTE；本仓库先支持离线指标与可视化，后续可扩展到完整评估）

---

## 1) 环境与依赖

### 1.1 Tested environment（本仓库已在如下环境中运行）
- torch==2.9.1+cpu
- torchvision==0.24.1+cpu

可用以下命令检查：
```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

> 说明：论文通常不会提供 `requirements.txt`。  
> 本仓库的 `requirements.txt` 是为了让复现代码可运行而维护的依赖清单。

### 1.2 安装依赖
在仓库根目录执行：
```bash
pip install -r requirements.txt
```

---

## 2) 仓库结构

```text
TEA-LAB/
├─ configs/
│  ├─ bc/                 # 行为克隆配置（后续阶段）
│  ├─ cae/                # CAE 训练配置
│  └─ cleaning/           # PCC 平滑/阈值等配置
├─ data/
│  ├─ raw/                # 原始数据（不改动）
│  │  └─ 16000clean7dirty.pkl
│  ├─ processed/          # 由脚本生成的混合数据 npz（训练/打分更快）
│  │  ├─ raindrop_mix.npz
│  │  ├─ plastic_mix.npz
│  │  └─ hitwall_mix.npz
│  └─ splits/             # 固定抽样索引（保证复现一致）
│     ├─ raindrop_mix_idx.npz
│     ├─ plastic_mix_idx.npz
│     └─ hitwall_mix_idx.npz
├─ notebooks/             # 调试/可视化（可选）
├─ outputs/               # 注意：本仓库 outputs/ 仅保留下面四个目录
│  ├─ cae/                # ckpt / log
│  ├─ scores/             # pcc/smooth/pred 等 csv
│  ├─ cleaned/            # clean mask / cleaned npz
│  └─ figs/               # 曲线图、可视化
├─ scripts/               # 命令行入口：每个阶段一个脚本
└─ src/                   # 核心实现（可 import）
   ├─ anomaly/            # pcc / smoothing / threshold / metrics
   ├─ datasets/           # pkl读取 / mix构造 / torch Dataset
   ├─ losses/             # reconstruction + latent_reference
   ├─ models/             # CAE + controller（后续）
   ├─ postprocess/        # 生成 clean mask / 导出 cleaned 数据
   ├─ train/              # 训练循环
   └─ utils/              # seed / io / logger
```

---

## 3) 数据说明

### 3.1 原始 pkl
`data/raw/16000clean7dirty.pkl` 是一个 dict：
- keys: `clean`, `foggy`, `greenmarker`, `plastic`, `raindrop`, `hitwall`, `debris`, `dirtytrain`
- values: `uint8` 图像数组，shape 为 `(N, 224, 224, 3)`，像素范围 0~255

### 3.2 混合数据（用于 Table II 风格评估）
混合数据按 **clean : dirty = 10 : 1** 构造并保存为：
- `data/processed/<dirty>_mix.npz`
- `data/splits/<dirty>_mix_idx.npz`（固定抽样索引，复现一致性的关键）

> 注意：`data/splits/*.npz` 不要删除，否则无法保证每次运行使用同一批 mix 数据。

---

## 4) 复现流程（推荐顺序）

> 所有命令都从仓库根目录执行：`TEA-LAB/`  
> 不确定脚本参数时：`python scripts/<xxx>.py -h`

### Step 0 — 检查 pkl 结构
```bash
python scripts/inspect_pkl.py --pkl data/raw/16000clean7dirty.pkl
```

### Step 1 — 构造 mix 数据集（clean : dirty = 10 : 1）
生成：
- `data/processed/<dirty>_mix.npz`
- `data/splits/<dirty>_mix_idx.npz`

```bash
python scripts/build_mix_dataset.py -h
python scripts/build_mix_dataset.py
```

> 说明：如果 `data/processed/` 和 `data/splits/` 中已经存在对应的 mix 与 idx 文件，可跳过本步进入 Step 2。

### Step 2 — 训练 CAE（含 latent reference loss）
输出写入：
- `outputs/cae/`

```bash
python scripts/run_train_cae.py -h
python scripts/run_train_cae.py
```

### Step 3 — PCC 打分 + 平滑 + 阈值
生成：
- `outputs/scores/*.csv`（包含 pcc / pcc_smooth / pred_anom 等）
- `outputs/figs/*.png`（PCC 曲线、阈值可视化）

```bash
python scripts/run_score_pcc.py -h
python scripts/run_score_pcc.py
```

### Step 4 — Table II 风格评估（Precision / Recall）
使用 mix 数据的 GT dirty 标签计算 Precision / Recall。

```bash
python scripts/run_eval_table2.py -h
python scripts/run_eval_table2.py
```

### Step 5 — 导出 cleaned 数据 / clean mask
输出写入：
- `outputs/cleaned/`

```bash
python scripts/run_clean_dataset.py -h
python scripts/run_clean_dataset.py
```

### Step 6 — 行为克隆（后续阶段）
待拿到 steering/actions 标签后：
```bash
python scripts/run_train_bc.py -h
python scripts/run_train_bc.py
```

---

## 5) outputs 约定（必须遵守）

本仓库 **outputs/** 只保留以下四个子目录：
- `outputs/cae/`：CAE checkpoints / logs  
- `outputs/scores/`：PCC 分数与预测（csv）  
- `outputs/cleaned/`：clean mask / cleaned 数据  
- `outputs/figs/`：曲线图、可视化  

---

## 6) 复现注意事项

- **复现一致性**：`data/splits/*_mix_idx.npz` 是关键（别删）。
- **改超参不要覆盖基线**：建议新建 config 文件，例如 `configs/cae/raindrop_v2.yaml`。
- CPU 训练较慢：可先用 debug_small 配置做快速验证，再跑完整配置。
