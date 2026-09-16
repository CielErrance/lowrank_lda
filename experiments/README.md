# 实验归档（探索/负结果，生产管线不依赖）

| 文件 | 实验 | 结论 | worklog |
|---|---|---|---|
| lr_fg10_rn50.py | FG10 RN50 backbone | RN50 标签质量瓶颈失效 | 2026-08-29_rn50_backbone |
| lr_dota3_rn50.py | ImageNet-C RN50 对照 | C 上 RN50 仍 15/15 全正 | 2026-08-29_dota3_rn50_official |
| extract_aug63_imagenet_c.py | C 域 aug63 特征提取 | 增强在 C 全面负收益（噪声域 baseline 腰斩） | 2026-08-30_imagenet_c_lr_vit |
| extract_multi_imagenet.py / run_multi_update.py | 6-update 多视图 update | −0.22 无收益（视图方差=噪声） | 2026-08-28（multi-update 节） |
