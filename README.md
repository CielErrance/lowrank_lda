# ELD-TTA：显式低秩 LDA 的测试时自适应

本仓库是论文方法 **ELD-TTA**（代码中类名 `LowRankLDA`）的完整实现与实验脚本。

**方法**：在 CLIP 冻结特征流上在线维护一个共享类内协方差的高斯判别器
Σ = AAᵀ + σ²I —— A 由 **Frequent-Directions（FD）流式草图**（分块 SVD）学到，
中心化用 Welford 无偏散度增量，打分走 **Woodbury 恒等式**（永不构造 D×D 矩阵）。
存储 O((k+1)·D)，判别器延迟与类别数无关。

## 主要结果

| 基准 | 配置 | AVG |
|---|---|---|
| **ImageNet 5 域**（imagenet/v2/sketch/a/r） | aug63 + DPE prompt + FD r=64, l=k + hard 分配 + 乘性融合 | **66.47** |
| ↳ 对照：ADAPT（本机复现）/ 论文 | — | 66.45 / 66.53 |
| ↳ 对照：DOTA（ADAPT prompt 复跑） | — | 65.65 |
| FG10（10 域，soft 更新） | FD r=64, l=k / r=256, l=k | 72.16 / **72.66** |
| ImageNet-C（15 域 severity-5，单视图） | online r=64（ViT / RN50） | 29.38 / 14.02 |

ImageNet 5 域逐域（我们的 66.47 vs ADAPT 自跑）：imagenet **72.08** vs 70.93 ·
v2 **64.96** vs 64.43 · sketch 52.02 vs 53.12 · a 63.01 vs 63.07 · r 80.30 vs 80.70。

## 文件

| 文件 | 作用 |
|---|---|
| `method.py` | **`LowRankLDA`（核心）**：FD 草图 + block FD 遗忘 + Welford 中心化 + Woodbury 打分 + streaming 软标签 μ + 可选文本先验锚定；`sigma_shrink` 支持 iso/add/scale/trace/diag |
| `data.py` | FG10 特征缓存加载与结果 CSV 写入 |
| `datasets.py` | FG10 数据集加载器（CoOp/Zhou 布局，bs=1 流序可复现） |
| `extract_features.py` | FG10 单视图特征提取 → `feats/<ds>.npz` |
| `fg10_online.py` | FG10 流式 runner（update-then-predict；`--hard_weight` 支持 soft/cos/one/prob） |
| `fg10_transductive.py` | FG10 transductive runner（全量 update → 统一打分） |
| `imagenet_online.py` | ImageNet 系 runner：`--family inet5`（5 域，融合/hard/门限/μ 锚定全开关）与 `--family inc`（ImageNet-C，含 `--mode extract` 提特征） |
| `imagenet_transductive.py` | ImageNet-C transductive（`--backbone` × `--cache`） |
| `aggregate.py` / `run_all.sh` | FG10 多 GPU 分片 + 合并 |
| `analysis/` | 机制诊断脚本（低秩 vs 全秩、ADAPT 估计器移植、DPE prompt 提取），见其 README |
| `experiments/` | 负结果与旁支实验归档（RN50、multi-update、ImageNet-C aug63） |
| `class_feat/` `clip/` | FG10 文本原型（来自 ADAPT 项目的 GPT 描述嵌入）/ vendored OpenAI CLIP |
| `feats/` | FG10 特征缓存（gitignore，缺省时由 `extract_features.py` 生成） |

## 复现命令

```bash
PY=/path/to/python        # torch + numpy + tqdm（见 requirements.txt）
cd lowrank_selftrain

# ImageNet 5 域（66.47 配方）：aug63 缓存 + DPE 文本原型 + hard 分配 + 乘性融合
#   ADAPT_ROOT / DPE 原型路径见下节环境变量；DPE 原型用 analysis/extract_dpe_prompts.py 生成
$PY imagenet_online.py --family inet5 --gpu 0 --cache aug63 --rank 64 --sketch 1 \
    --shrink add --sigma_eps 0.05 --fuse adaptmul --adapt_scale 20 \
    --hard_assign --hard_weight prob --pth_tier GPT_w_DPE_class_emb

# 同基准的加性融合对照（65.96~66.35 一档，见 work_logs）
$PY imagenet_online.py --family inet5 --gpu 0 --cache aug63 --fuse dotaclip

# FG10（r=64 用 --rank 64）
$PY fg10_online.py --rank 256 --sketch 1 --gpu 0

# ImageNet-C：提特征 → online / transductive
$PY imagenet_online.py --family inc --mode extract --backbone vit --gpu 0
$PY imagenet_online.py --family inc --backbone vit --cache singlev --rank 64 --gpu 0
$PY imagenet_transductive.py --backbone vit --cache singlev --rank 64 --gpu 0
```

## 环境变量（路径配置）

脚本默认使用本机绝对路径，其他机器用环境变量覆盖即可，无需改代码：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `ADAPT_ROOT` | `/home/liangyiwen/wufan/ADAPT` | 文本原型 `pre_extracted_class_feat/`、aug63 特征 `feats/` 所在目录 |
| `IMAGENET_FEATS_DIR` | `/home/liangyiwen/wufan/.imagenet_feats` | imagenet 单视图特征缓存 |
| `IMAGENET_C_DATA` / `IMAGENET_C_OUT_*` | `/home/liangyiwen/datasets` 等 | ImageNet-C 数据根与特征输出目录 |
| `FG10_FEATS_DIR` | `<repo>/feats` | FG10 特征缓存 |

aug63 特征提取脚本（imagenet 家族 63 视图 + 低熵 top-10% 聚合）位于 `ADAPT/extract_aug63_features.py`（本库外）。

## 文本原型档位

`--pth_tier` 选择文本原型目录（默认 `GPT_w_Custom_class_emb`）。切换非默认档时 runner 会用
`100·feats@W` 重算 `clip_ll`，保证软标签/融合与所选原型一致。本工作比较过的档位：

| 档位 | 构造 | 5 域零样本 |
|---|---|---|
| Custom（ADAPT GPT 描述 × 每域单模板） | 描述句 + 模板 | 61.80 |
| Ensemble（7 通用模板 × 描述） | 模板集成 | 61.96 |
| Img_temp（80 模板 × 描述） | 模板集成 | 61.88 |
| **DPE**（7/80 模板 + CuPL 描述，来自 DPE-CLIP） | 模板 + 描述 | **62.38** |
| Mix（逐域选档） | — | 62.27 |

## 关键结论（详见 `wufan/work_logs/`）

1. **低秩不损失判别力**：r=64 与全秩 512 在 soft/oracle 两种标签口径下持平（≤0.12pp）；
   协方差容量不是任何差距的来源（`analysis/oracle_lr_vs_full.py`）。
2. **ADAPT 的分数优势 ≈ 全部来自 bank 记忆结构**，而非 KS-ridge 估计器：把 KS-ridge 放到
   我们的全流协议上（去 bank）得 60.36，反而低于我们的 soft 流式 61.01
   （`analysis/adapt_pure_gda.py`）。
3. **hard 分配（质量集中到 pred 类）可免费回收 bank 优势的 87%**（纯分数 61.01→64.15 vs
   bank 的 64.57），配 hard 更新时三种权重（cos/概率/1）等价。
4. **hard 的价值在乘性融合下才释放**：加性融合下 hard 仅 +0.01~0.04，乘性下 +0.53；
   soft 训练的统计量跨类涂抹使 log_softmax 后验形状失真，乘性调制无从发挥。
5. **prompt 档最强的是 DPE（模板 + CuPL 描述双管齐下）**；但零样本排名 ≠ 下游质量——
   逐域按零样本选档反而更差（66.35 < 66.48）。
6. **hard 分配的收益是域相关的**：ImageNet 系（1000 类大流量）+3.14 纯分数，FG10
   （10~397 类）−1.63~−2.21——soft 的跨类分摊是小数据域的样本效率来源。
7. **transductive vs online**：FG10 上 transductive 略优（+0.5~1.0）；ImageNet-C 上二者相当。
8. **增强**：63 视图 + 低熵选择在 ImageNet 系 +4.7；在 ImageNet-C 上负收益（损坏图像上
   熵选择反向）。
