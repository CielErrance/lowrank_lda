# analysis/：诊断与对照脚本

这些脚本支撑论文中的机制分析（不是生产管线；生产 runner 在上一级目录）。
全部假设 aug63 特征缓存与文本原型位于 `ADAPT_ROOT`（见仓库根 README 的环境变量说明）。

| 脚本 | 回答什么问题 | 关键结论 |
|---|---|---|
| `oracle_lr_vs_full.py` | 低秩 r=64 与全秩 512 判别器在干净标签下是否等价？ | 两者持平（≤0.12pp，soft 与 oracle 标签两种口径）→ 低秩截断不损失判别力；另附 batch ceiling 参考 |
| `adapt_pure_gda.py` | ADAPT 的分数优势来自 KS-ridge 估计器还是 bank 记忆结构？ | bank 贡献 ≈全部：去 bank 后 KS-ridge（60.36）低于我们 LR（61.01）；hard 分配可回收 87% 的 bank 优势 |
| `extract_dpe_prompts.py` | 用 DPE-CLIP 的 CuPL prompt 档替换文本原型 | DPE 档（模板+CuPL 描述）是四档中最强（零样本 +0.58），端到端把 5 域推到 66.47 |

## 用法

```bash
PY=/path/to/python        # 需要 torch/numpy/tqdm + vendored clip
cd lowrank_selftrain

# 低秩 vs 全秩（5 域，纯判别器；全秩部分较慢）
$PY analysis/oracle_lr_vs_full.py --gpu 0

# ADAPT 估计器移植：ks（bank+KS-ridge） / ks_nobank（去 bank） / lr（我们）
$PY analysis/adapt_pure_gda.py --gpu 0 --mode ks_nobank --domains imagenetv2 imagenet_sketch

# 提取 DPE 文本原型（5 域可并行，一域一进程）
for ds in imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r; do
  $PY analysis/extract_dpe_prompts.py --gpu $N --domains $ds
done
```

## 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `ADAPT_ROOT` | `/home/liangyiwen/wufan/ADAPT` | 文本原型（`pre_extracted_class_feat/`）与 aug63 特征（`feats/`）所在目录 |
| `AUG63_FEATS_DIR` | `$ADAPT_ROOT/feats` | aug63 缓存目录 |
| `PTH_ROOT` | `$ADAPT_ROOT/pre_extracted_class_feat/ViT-B16` | 文本原型根目录 |
| `DPE_ROOT` | `/home/liangyiwen/wufan/DPE-CLIP` | DPE-CLIP 仓库（提供 CuPL prompts 与各域模板） |
| `DPE_OUT_DIR` | `/mnt/newdata/liangyiwen/clip_class_emb/ViT-B16/GPT_w_DPE_class_emb` | DPE 原型输出目录 |
