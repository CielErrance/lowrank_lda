# lowrank_selftrain：显式低秩 LDA + 自训练（自包含，可复现）

把 **low-rank LDA + self-train (n_ref=200)** 从 `online_softlabel` 抽出为**自包含、可独立运行**的方法
（**非** auto / 归一化熵门控版本）。模型假设共享类内协方差 $\Sigma = AA^\top + \sigma^2 I$，
用 Frequent-Directions 流式草图 $S\,[l,D]$（分块 SVD）在线估计；自训练为在线 EM，
用 LDA 判别替代 CLIP 软标签更新每类均值，配每类可靠性门控 $w_c = \mathrm{clamp}(N_c/n_{\mathrm{ref}},0,1)$。
全部关键公式与原理见 `online_softlabel/doc/low_rank_lda_and_self_train.md` §3–§4。

Clone 后即可运行：CLIP 权重首次使用自动下载到 `~/.cache/clip`；只需提供原始数据集（见下）。

## 文件

| 文件 | 作用 |
|---|---|
| `method.py` | `LowRankLDASelfTrain` 类：FD 草图 + block FD + Woodbury 求逆 + 自训练（streaming/selftrain） |
| `data.py` | 数据加载：`feats/` 特征缓存 + 打包的文本原型 `class_feat/` |
| `datasets.py` | FG10 数据集加载器（Zhou/CoOp 目录布局，自 `fewshot_datasets.py` 移植） |
| `extract_features.py` | 特征提取：原始图片 → `feats/<dataset>.npz`（CLIP ViT-B/16, bs=1） |
| `run.py` | 主运行脚本：bs=1, update-then-predict，无标注无训练数据；缺缓存时自动提取 |
| `aggregate.py` | 合并各 GPU 分片 CSV → `results_lowrank_selftrain_all.csv` |
| `run_all.sh` | 一键：5 GPU 分片跑 FG10 + 自动聚合 |
| `class_feat/GPT_w_Custom_class_emb/` | 打包的 CLIP 文本原型（10 个 `.pth`，与实验完全一致） |
| `clip/` | vendored OpenAI CLIP（tokenizer + model） |
| `requirements.txt` | 运行时依赖 |

## 安装

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 数据准备

需要 10 个数据集的**测试划分**原始图片 + Zhou 划分 JSON。每个数据集放在
`<data_root>/<dirname>/` 下（布局见 `datasets.py` 的 `ID_TO_DIRNAME` 与 `_PATH_SPEC`），
如：

```
<data_root>/
├── fgvc_aircraft/  (variants.txt, images_variant_test.txt, images/)
├── caltech-101/    (101_ObjectCategories/, split_zhou_Caltech101.json)
├── stanford_cars/  (split_zhou_StanfordCars.json, ...)
├── dtd/            (images/, split_zhou_DescribableTextures.json)
├── eurosat/        (2750/, split_zhou_EuroSAT.json)
├── oxford_flowers/ (jpg/, split_zhou_OxfordFlowers.json)
├── food-101/       (images/, split_zhou_Food101.json)
├── oxford_pets/    (images/, split_zhou_OxfordPets.json)
├── sun397/         (SUN397/, split_zhou_SUN397.json)
└── ucf101/         (UCF-101-midframes/, split_zhou_UCF101.json)
```

即 CoOp / CLIP 标准 few-shot 划分（`split_zhou_*.json`）。下载方式参见
[vMFcache 的 `datasets/download_datasets.sh`](../vMFcache/datasets/download_datasets.sh)
或 [CoOp](https://github.com/KaiyangZhou/CoOp)。

## 特征提取（一次性）

```bash
python extract_features.py --data_root /path/to/datasets --gpu 0
```

对每个数据集提取 CLIP ViT-B/16 特征到 `feats/<dataset>.npz`（bs=1，stream 顺序固定）。
CLIP 权重首次自动下载（约 354MB）。提取后可删除原始图片目录以节省空间。

## 运行

```bash
./run_all.sh                          # 5 GPU 并行跑 FG10 + 聚合
DATA_ROOT=/path/to/datasets ./run_all.sh   # 若未提取特征，自动提取后运行
```

单 GPU / 单数据集调试：

```bash
python run.py --gpu 0 --datasets dtd
python run.py --gpu 0 --datasets dtd --data_root /path/to/datasets   # 缺缓存自动提取
```

## 超参数（`run.py`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--rank` | 64 | 低秩因子 $A$ 的秩 $k$ |
| `--sketch` | 2 | FD 草图高度 $l=\mathrm{sketch}\cdot k$（2 → $l=2k$ 需二次 eigh；1 → $l=k$ 免分解） |
| `--mu` | `selftrain` | 均值更新：`selftrain`（在线 EM）或 `streaming`（CLIP 软标签） |
| `--nref` | 200 | 每类可靠性阈值 $n_{\mathrm{ref}}$ |
| `--temp` | 2.0 | 自训练 E 步 LDA softmax 温度 |
| `--warmup_frac` | 1.0 | warmup 样本数 = `frac × C`，之前用 CLIP 兜底 |

## 复现

使用打包的文本原型（`class_feat/`，与实验一致）与固定 stream 顺序（seed 0 + shuffle），
FG10 上应复现（简单平均）：

- low-rank LDA + selftrain **n_ref=200**：**72.39**（超全协方差 LDA 72.15 约 +0.24）
- low-rank LDA 纯流式：71.65
- 分数据集：eurosat +5.29（CLIP 弱，全信 LDA）、food101 −2.62（CLIP 强，已知残留缺口）
