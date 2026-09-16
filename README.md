# lowrank_lda：显式低秩 LDA（最终代码库）

生产方法：**LowRankLDA** —— 共享类内协方差 Σ = AAᵀ + σ²I（Frequent-Directions
流式草图，分块 SVD），Welford 无偏中心化，Woodbury 求逆；μ 为 streaming 软标签加权平均
（CLIP 软标签），可选文本原型锚定（mu_prior/mu_alpha）。
核心公式与原理见 `online_softlabel/doc/low_rank_lda_and_self_train.md` §3–§4。

三条生产管线（数据集 → runner）：

| 管线 | 数据集 | runner | 生产配方（5/10 域 AVG） |
|---|---|---|---|
| FG10 流式 | FG10 10 域 | `fg10_online.py` | r=256/l=k streaming → **72.66**；trans 73.60 |
| FG10 transductive | FG10 10 域 | `fg10_transductive.py` | r=64/l=k → **73.19**（r256 73.60） |
| ImageNet 系 online | imagenet/v2/sketch/a/r | `imagenet_online.py --family inet5` | aug63 + dotaclip + add ε=0.05 → **65.71**（ε=0.01 → 65.78） |
| ImageNet-C online | 15 域 severity5 | `imagenet_online.py --family inc` | 单视图 online r64 → **29.38**（ViT）/ 14.02（RN50） |
| ImageNet-C trans | 15 域 severity5 | `imagenet_transductive.py` | 单视图 trans r64 → **29.26**（ViT）/ 13.84（RN50） |

对照基线：CLIP zero-shot 68.25 / 65.61 / 27.06；ADAPT（原版复现）66.45（imagenet 系）。
全部实验日志与结论见 `wufan/work_logs/`。

## 文件

| 文件 | 作用 |
|---|---|
| `method.py` | `LowRankLDA`：FD 草图 + block FD + Woodbury + streaming 软标签 μ + 可选 mu_prior/mu_alpha 文本先验锚定 |
| `data.py` | FG10 特征缓存加载（`feats/` + `class_feat/` 文本原型） |
| `datasets.py` | FG10 数据集加载器（Zhou/CoOp 布局，bs=1 流序可复现） |
| `extract_features.py` | FG10 单视图特征提取 → `feats/<ds>.npz` |
| `fg10_online.py` | FG10 流式 runner（bs=1 update-then-predict，缺缓存自动提取） |
| `fg10_transductive.py` | FG10 transductive runner（全量 update → 统一打分） |
| `aggregate.py` / `run_all.sh` | FG10 多 GPU 分片 + 合并 |
| `imagenet_online.py` | `--family inet5`（5 域，融合全家桶/μ 锚定/门限/hard 开关）与 `--family inc`（C 流式 + `--mode extract` 提特征） |
| `imagenet_transductive.py` | ImageNet-C transductive（全量 update → 统一打分；`--backbone` × `--cache`） |
| `experiments/` | 实验/负结果归档（RN50 探索、C 域 aug63、multi-update；各带结论指针） |
| `results/` | 历史 CSV 结果 |
| `class_feat/` `clip/` `feats/` | 文本原型 / vendored CLIP / FG10 特征缓存 |

## 生产复现命令

```bash
PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python

# FG10 流式（r=256/l=k 生产）—— r=64 用 --rank 64
$PY fg10_online.py --rank 256 --sketch 1 --gpu 0

# FG10 transductive
$PY fg10_transductive.py --rank 64 --sketch 1 --gpu 0

# ImageNet 系 5 域（65.71 配方默认值；负结果开关 --fuse alphamix/adaptmul/--gate/--hard_assign 保留供复现）
$PY imagenet_online.py --family inet5 --fuse dotaclip --cache aug63 --sketch 1 \
    --shrink add --sigma_eps 0.05 --mu_alpha 0 --gpu 0

# ImageNet-C 特征提取（单视图 ViT；RN50 加 --backbone rn50）
$PY imagenet_online.py --family inc --mode extract --backbone vit --gpu 0

# ImageNet-C online / trans（单视图 online r64 生产；shrink 默认按 family=iso）
$PY imagenet_online.py --family inc --backbone vit --cache singlev --rank 64 --gpu 0
$PY imagenet_transductive.py --backbone vit --cache singlev --rank 64 --gpu 0
```

aug63 特征提取（imagenet 家族）：`ADAPT/extract_aug63_features.py`（本库外，依赖 ADAPT 数据根）。

## 关键实验结论（详见 work_logs）

1. **协议**：transductive ≥ online 仅 FG10 成立（+0.5~1.0）；ImageNet-C 上 online≈trans
   （online 的表面优势 = self-inclusion 启动暂态，非统计优势）。
2. **rank**：小流量（FG10）r=256 > r=64；大流量（ImageNet-C）r=64 ≥ r=256——样本量决定
   低秩饱和点。
3. **融合**：dotaclip（clip + w·lda 小尺度加性）+4.68；对称凸组合（alphamix）/乘性调制
   （adaptmul）均无增益——LDA 分数只能当加性修正项。
4. **增强**：aug63 在 ImageNet 系 +4.7；在 ImageNet-C 负收益（损坏图像熵筛反向选择）。
5. **vs ADAPT**：65.71 vs 66.45 的 0.74 差距 = KS-ridge bank 分数在乘性调制下的优势
   （85%）+ 流程耦合（15%）；乘性下 rank 64/128/256 无差异——根因是 FD 残差各向同性
   假设 vs bank 全谱协方差的表示差，LR 框架内不可弥合（1/8 参数换 0.7 是既定交换）。
