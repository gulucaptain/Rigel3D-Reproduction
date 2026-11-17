# Anymate 真实子集：实现、差异和实验核查

审阅：2026-11-16。数值完整表见 [REPORT](../results/round1/REPORT.md)，全部种子与误差口径见 `results/round1/summary.json`。**这是基于真实资产的局部解码器训练与留出重建实验，不是完整 Rigel3D，也未复现原论文表格。**

## 数据接入和独立划分

固定官方 Hugging Face 数据版本 `096e25372f38db1964c5f5ecaa68c45d78f72ec8`。仅使用官方 train_0 的资产建立训练/验证子集，测试仅来自官方 test。按 `sha256("rigel-subset-v1:"+asset_name)` 排序，先取训练 96、验证 16；官方 test 独立取 32。没有按结果或几何复杂度挑样本。完整资产 ID 和逐存储块校验写入 `anymate_subset_manifest.json`。

torch-save 实际是 ZIP。通过 HTTP range 读取目录与 `data.pkl`；元数据只允许特定 tensor-storage 描述，禁止任意 pickle global 执行，不调用外部 `torch.load(weights_only=False)`。只提取选择资产的 storage，并校验长度、ZIP CRC 与 SHA-256。张量提取传输约 51.1MB，目录/元数据另约 47.1MB；没有下载完整 10.8GB 训练分片或 4.7GB 测试包。部分读取只能验证这些成员，不声称校验了完整原始大包。

实际数据与 README 不同：`pc` 是 8192×6，`conns` 是父索引向量，蒙皮为 `skins_index` 与 `skins_weight` 稀疏表示；不是 8196 点、连接邻接矩阵或直接稠密权重。适配器严格保留点顺序、按索引累加权重，不裁剪错配。有效非零权重行归一化；零质量行明确排除蒙皮损失与评价，记录数量。

部分层级多根或成环。未修补成“看起来正确”的树：训练 83/96、验证 14/16 满足单根无环条件，测试 32/32 可用；其余资产仍参与蒙皮任务。15 个骨架排除原因逐项保存：14 个不满足单根，1 个有不连通循环。资产 ID 与原始来源家族在三个子集间均无重叠。原始骨头尾与 joint 集的对应、文件哈希、权重行和等已独立核查。

## 实现与条件信息

骨架 AR 解码器包含右移教师强制、可学习 BOS、EOS 分类、128³ 坐标头、父索引头、序列位置编码、因果自注意力以及依次访问表面/骨架上下文的 cross-attention。根父指自己，其他父索引只能指向较早关节；推理不喂真实前缀，只按预测 token 继续，记录 EOS 和 96 关节上限。

蒙皮解码器使用位置/法向、骨头尾坐标的 Fourier 位置编码、自注意力和双上下文注意力，再做点—骨相似度及骨维 softmax。变长 joint/bone/context 均有掩码。两任务共享小型可学习上下文投影，按骨架 CE + 蒙皮软 CE 联合训练。

**重要条件范围：** 此轮没有假装拥有 TRELLIS/DINO 训练特征。表面上下文为真实点云上的 16 维确定几何特征；骨架上下文来自 64³ 骨段体素，压到最多 48 个空间分布 token，再经学习投影。

- `rig`：按照 Eq.2 用真实蒙皮池化几何特征，编码真实骨架。这是含真实绑定的重建代理，不是独立 image-to-rig 推断。
- `geometry`：骨段上下文仅由给定真实骨段几何构造，不使用目标蒙皮权重池化；**仍有真实骨架几何输入**，不能称 mesh-only skeleton generation。
- 蒙皮评价提供同一真实骨骼列表，点/骨索引可对齐，所以直接比较权重。完整生成骨架与网格的评价需要最优传输等对应处理；本轮不声称已验证该部分。

## 实验设置和选择

每种设置种子 0/1/2，2000 步，batch=4，宽度 64、2 层、约 597K 参数；训练每资产抽 256 有效点，评价固定 512 点。AdamW lr=1e-4，200 warmup 后 cosine。归一化按点、joint、bone 合并包围盒的最长边统一到 1，所有几何共同变换。这是明确的重建约定，作者 TRELLIS 预处理/骨架包围盒约定尚未完全复刻。

各种子只按验证集联合损失选 checkpoint；测试不选种子、不调权重。报告三种子的均值和样本标准差，不是对完整数据集的置信区间。初期 20 步 smoke 单独保留，不计入 9 次正式训练。

原文约 225K 训练资产、4×48GB GPU、约 10 天的联合 AE，以及预训练 Sparse VAE/DINO 特征，均与此轮不同；5000 warmup 在此轮明确缩成 200。模型层数/维数、EOS 分类细节原文未充分披露，因此不把本地结构或训练时间视作作者配置。

## EOS 失败及修正对照

无权重 EOS CE 在稀少正例下，教师强制损失低却可始终预测“继续”：`rig` 和 `geometry` 都在 3×32 次测试中 **96/96 达到最大关节数**。这不是有效自由生成成功。

增加显式的停止类别平衡：继续/停止两类分别求 CE 均值，再各取 0.5。其余数据/种子/预算不变，单独运行三种子，不覆盖原结果。达到上限降至 **2/96**，但 J2J×100 从 **13.341±1.167** 变差到 **15.015±1.281**。只能说改善停止行为，不能说修复了骨架位置或拓扑；父索引的教师强制准确率也有下降。

GT 量化 oracle 的 J2J 约 0.00373；实际自由生成误差远大于此，因此不能归因于 128³ 量化精度。教师强制几何、自由生成和已知 GT 数量的截断诊断另存为不同字段；其中教师强制和 GT 数量截断使用额外目标信息，绝不当成主结果。

## 指标与对照

主报告采用固定 Anymate/RigNet 源码的距离约定：未平方欧氏距离、双向均值/2，骨段按 `round(length/0.005)` 均匀采样。初版运行日志中的解析 J2B 与每骨等权 32 点 B2B 保留为 v1 诊断，主比较重新从保存预测计算为 `evaluation_metrics_v2.json`。另报告四倍采样密度敏感性，避免把不同定义混用。

蒙皮 L1 同时保存逐元素平均和每点向量 L1，L2 向量范数与逐元素 MSE 分开。Precision/Recall 使用官方阈值 0.05 和逐点宏平均。额外训练骨段距离 softmax 温度只在验证集选出 0.02；形状检索对照只在训练资产中按表面 Chamfer 选择骨架，不访问测试骨架。它们是本地对照，不冒充 Anymate/Puppeteer/AniGen 原始基线。

原设置蒙皮逐元素 L1×100 为 **3.484±0.084**，几何对照 **3.490±0.070**，骨段距离对照 **3.501**；优势很小且跨种子并不稳定。训练骨架检索 J2J×100 为 **10.015**，比当前神经解码器更好。负结果完整保留，不宣称匹配论文。

## 完整生成流程的实际推进

已追溯所有 112 个训练/验证家族到 Anymate 官方原资产映射；其中 101 个可在固定 Objaverse-XL GitHub 索引中定位。首个训练资产的原始 FBX 已按 SHA-256 核验，Blender 5.2.3 在禁用自动脚本的后台模式完成四个 224² 视图，官方 DINOv2 ViT-S/14 在 MPS 提取 4×256×384 特征。

首次导出未求值的 FBX 顶点与实际渲染姿态不一致，导致可见顶点为零；失败记录保留。改为导出 armature/modifier 求值后的网格，并加入非空检查，现有 1871 顶点中 1253 被四视图观测，聚合为 1028 个 64³ 活跃表面体素，384 维特征。它是后续真实 Sparse VAE 的输入准备，**不是已经得到作者 SLats**，也未混入本轮九次训练。

目前没有训练/移植 FlexiCubes 几何解码、完整稀疏 VAE、四个图像条件 flow 或语义命名模块。完整生成训练仍需一致的网格/DINO/rig 预处理、预训练模块兼容性与联合重建验证；停止行为改善不足以支持直接宣称完整生成成功。原始资产、图像、点预测和模型权重保持本地，不上传到总览。

## 重建入口

```sh
python scripts/inspect_anymate.py --split train
python scripts/inspect_anymate.py --split test
python scripts/download_subset.py
python scripts/prepare_subset.py
python scripts/audit_data.py
python scripts/train_subset.py --mode rig --device mps --name round1
python scripts/train_subset.py --mode geometry --device mps --name round1
python scripts/train_subset.py --mode rig --device mps --name round1_balanced --balanced-stop
python scripts/audit_evaluations.py
python scripts/verify_checkpoint.py --suite round1
python scripts/verify_checkpoint.py --suite round1_balanced
python scripts/report_training.py
```

训练目录非空时拒绝覆盖；重跑另选 `--name` 并保留历史。下载缓存逐资产验哈希；prepare 可重建相同预处理，配置保存其清单哈希；回放拒绝清单不一致。

生成数据准备单独运行：

```sh
python scripts/resolve_original_mesh.py --download-index
/Applications/Blender.app/Contents/MacOS/Blender --background --disable-autoexec --python scripts/render_preflight.py
python scripts/extract_preflight_features.py
```

此额外流程首次需要约 0.8GB 公共网格元数据索引和约 86MB DINO 权重；已缓存可复用。Blender/MPS 在受限后台不可用时须使用正常桌面执行环境。完整依赖版本写入 `requirements-lock.txt`。
