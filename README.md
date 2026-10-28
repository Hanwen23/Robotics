# Robotics：实验一 · 视触主动感知

本仓库实现课程实验一的最小可复现流程：通过真实视觉与触觉视频训练小型分类模型，比较三种模态输入，回放随机／主动接触选择，并验证 MuJoCo 物理接触与 TACTO 光学触觉渲染。

当前实现用于环境验收和后续实验开发。主动接触采用**离线候选回放**，仿真采用**球面／平面几何压入适配**，尚未组成机器人闭环操作系统。

## 目录

- [功能与验证范围](#功能与验证范围)
- [环境要求](#环境要求)
- [从零运行](#从零运行)
- [仓库结构与上传范围](#仓库结构与上传范围)
- [数据来源与处理](#数据来源与处理)
- [模型与主动接触流程](#模型与主动接触流程)
- [实验参数](#实验参数)
- [输出文件](#输出文件)
- [已有最小测试结果](#已有最小测试结果)
- [常见问题](#常见问题)
- [后续实验方向](#后续实验方向)
- [来源与第三方许可](#来源与第三方许可)

## 功能与验证范围

| 路径 | 已实现内容 | 最小检查 |
| --- | --- | --- |
| 数据 | 固定版本下载、文件大小和 SHA256 校验、视频解码、时间戳配对 | 真实配对图像可读取，对齐误差不超过 100 ms |
| 分类 | 双分支 CNN，仅视觉／仅触觉／视触融合三个分类头 | 损失与梯度有限，优化器更新参数，保存／重载结果一致 |
| 接触选择 | 随机选择、基于视觉特征差异的主动选择、触觉更新与停止 | 在同一测试录像候选池执行两种策略，输出选择轨迹 |
| 仿真 | MuJoCo 球体与平面接触，TACTO 根据压入深度渲染 | 正接触力、正压入深度、接触前后光学图像变化、场景 RGB 渲染 |

不需要预训练模型、Hugging Face Token、CUDA 或机器人硬件即可运行默认测试。

## 环境要求

已验证的平台为 macOS / Apple Silicon arm64，Python 3.12.14。默认训练使用 CPU；本机 MPS 已单独通过卷积前向和反向计算检查，完整训练基线仍以 CPU 为准。

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.12，已验证补丁版本 3.12.14 |
| PyTorch / torchvision | 2.6.0 / 0.21.0 |
| MuJoCo | 3.3.7 |
| TACTO 渲染源代码 | 0.0.3 的独立渲染模块 |
| NumPy / SciPy | 1.26.4 / 1.14.1 |
| pyrender / pyglet / PyOpenGL | 0.1.45 / 1.5.31 / 3.1.0 |
| OpenCV | opencv-python-headless 4.11.0.86 |

其他要求：Git、curl，以及可访问的图形显示会话。建议至少预留 2 GB 空间给数据、依赖和生成文件。

`requirements.txt` 记录主要依赖，`requirements.lock.txt` 固定完整依赖；安装脚本使用锁定文件和清华 PyPI 镜像。其他操作系统尚未完成验证，尤其需要另行配置 OpenGL 后端。

## 从零运行

### 1. 克隆仓库

```bash
git clone git@github.com:Hanwen23/Robotics.git
cd Robotics
```

没有配置 GitHub SSH 的机器可以通过公开 HTTPS 地址克隆：

```bash
git clone https://github.com/Hanwen23/Robotics.git
cd Robotics
```

### 2. 创建环境并安装依赖

先准备 Python 3.12，再运行：

```bash
bash scripts/setup.sh
```

脚本优先复用项目中的 `.venv`。创建新环境时，依次检查 `PYTHON_BIN`、PATH 中的 `python3.12` 和本机 Codex 自带的 Python。它不会改动系统 Python 或全局 pip 设置。

指定解释器的例子：

```bash
PYTHON_BIN=/opt/homebrew/bin/python3.12 bash scripts/setup.sh
```

后续命令直接调用 `.venv/bin/python`，不要求激活环境。

### 3. 下载真实视触数据

```bash
.venv/bin/python scripts/download_data.py
```

默认下载约 **594.95 MB**，包含 **22 段录像、88 个文件**。这是 Touch and Go 的社区 probe 子集，不是原始完整数据集。下载不需要登录或 Token。

- 实际视频下载通过 `hf-mirror.com`；`curl --noproxy '*'` 强制直连，保留 TLS 验证。
- 固定仓库版本，逐文件校验大小与上游 LFS SHA256。
- 默认并发 4；降低并发可使用 `--workers 2`。
- 已存在且通过校验的文件会跳过。下载失败后可直接重跑；未完成文件保留为 `.part`，重跑时从该文件开头重新下载，不提供字节级续传。
- 本仓库附带必要的文件索引和标签，因此正常克隆后下载不依赖 GitHub Raw。若手动删除标签，脚本会尝试从固定版本的 GitHub Raw 补齐。

下载完成后生成或更新 `data/download_manifest.json`。原始数据存储于 `data/raw/touch_and_go/`。

### 4. 一键最小测试

```bash
bash scripts/run_smoke.sh
```

该命令检查依赖，并依次执行数据准备、12 个 epoch 的训练、两种接触策略回放、物理接触与光学渲染，最后生成中文报告。数据下载完成后，此流程不需要联网。

先打开 `outputs/实验一测试报告.md` 查看结果；终端最后应输出 `All smoke checks passed`。

### 5. 按步骤运行

```bash
# 提取配对图像并按录像划分数据
.venv/bin/python scripts/prepare_data.py

# 训练、分类对照、接触回放和模型保存
.venv/bin/python scripts/smoke_test.py

# MuJoCo 接触与 TACTO 渲染
.venv/bin/python scripts/simulation_smoke.py

# 根据真实输出生成报告，需要前述输出和下载清单
.venv/bin/python scripts/write_report.py
```

## 仓库结构与上传范围

```text
Robotics/
├── README.md
├── .gitignore
├── .gitattributes                 # 脚本换行与上游标签字节保留
├── requirements.txt
├── requirements.lock.txt
├── configs/
│   ├── smoke.yaml                 # 训练及接触回放参数
│   └── contact_scene.xml          # MuJoCo 最小物理场景
├── scripts/
│   ├── setup.sh                   # 独立环境安装与依赖检查
│   ├── download_data.py           # 固定版本镜像下载与校验
│   ├── prepare_data.py            # 解码、时间配对、按录像划分
│   ├── smoke_test.py              # CNN、对照、接触回放、检查点
│   ├── simulation_smoke.py        # 物理接触与 TACTO 渲染检查
│   ├── write_report.py            # 中文报告生成
│   └── run_smoke.sh               # 离线最小验证入口
├── references/
│   ├── SOURCES.md                 # 固定版本、来源与许可说明
│   ├── hf_dataset_tree.json       # 下载文件索引及上游摘要
│   ├── category_reference.txt     # 材质编号
│   ├── train_nointer_touch_instances.txt
│   ├── source_checksums.json      # 标签文件校验值
│   └── SeeingThroughTouch-LICENSE # 元数据来源仓库的 Apache 2.0 许可
├── vendor/tacto_renderer/         # 必需的上游渲染模块及 MIT 许可
└── data/download_manifest.json    # 文件路径、版本、大小与 SHA256
```

**上传内容**：可运行源码、实验配置、完整依赖锁定文件、小型必要元数据、下载清单及第三方许可。

**不上传内容**：`.venv/`、`.cache/`、原始视频 `data/raw/`、处理后的图像 `data/processed/`、模型权重和结果 `outputs/`、下载残片、Python 缓存、系统文件，以及临时查阅的完整上游代码、仓库树和 wheel。它们通过 `.gitignore` 排除，在已有工作目录中保留。

因此，刚克隆仓库时没有原始视频、训练好的模型或结果图，需要按前述步骤下载和生成。仓库中的下载清单用于复现和核对，不包含实际视频。

## 数据来源与处理

### 固定版本

| 内容 | 来源与版本 |
| --- | --- |
| 视觉／触觉原始数据 | `kingJulio/touch-and-go-probe`，`ba0bee2d863aabee6b3d533a4e15618d3d149ab1` |
| 接触区间与类别映射 | `kaistmm/SeeingThroughTouch`，`edd0e04430b966fcf08b8e17fe9955f71a2dcbed` |

原始数据每段录像的结构：

```text
data/raw/touch_and_go/<sequence_id>/
├── video.mp4       # 视觉视频
├── gelsight.mp4    # GelSight 触觉视频
├── time1.npy       # 视觉帧时间戳
└── time2.npy       # 触觉帧时间戳
```

标签文件采用 `video_id,start,end,category` 格式。准备脚本在每个接触区间等间距抽取 3 个视觉帧，再根据最近时间戳匹配触觉帧，将配对图像缩放到 96×96 保存。模型读取时进一步缩放为配置指定的 64×64。

当前原始子集中，有 18 段录像匹配训练元数据；Grass 只出现于一段录像，不能形成独立录像划分，因此不进入分类实验。其原始视频仍会下载。另 4 段录像不参与本次标签匹配。

最终分类使用 17 段录像、98 个接触区间、294 对图像，三个类别为 Concrete、Tree、Rock：

| 类别 | 训练图像对 | 测试图像对 |
| --- | ---: | ---: |
| Concrete | 6 | 12 |
| Tree | 3 | 3 |
| Rock | 186 | 84 |
| 合计 | 195 | 99 |

数据按**录像**划分，训练和测试录像交集为空。这是本项目的自定义最小划分，并未采用上游官方评测集。录像分离也不能保证物理对象分离。

`data/processed/samples.json` 保存每对图像的来源录像、接触编号、帧索引、类别、划分和时间误差。`summary.json` 保存整体统计；脚本在对齐误差超过 100 ms 时终止，首次验证的最大误差约为 17.17 ms。

## 模型与主动接触流程

### 分类模型

视觉和触觉分别使用一个轻量 CNN 编码器，每个编码器输出 48 维特征：

- 视觉特征进入仅视觉分类头。
- 触觉特征进入仅触觉分类头。
- 两路特征拼接为 96 维，进入融合分类头。

模型从头训练，不下载预训练权重。训练对三个分类头的交叉熵取平均，并用类别均衡采样缓解训练数据数量差异。

### 接触回放

每个测试录像作为一个回放 episode，同一录像中记录的不同接触构成候选池；每个接触取已抽取帧的中间帧作为代表。

1. 使用初始视觉特征计算类别概率。
2. 随机策略从剩余候选中随机选择；主动策略选择与已观察视觉特征最近距离最大的候选。
3. 只有候选被选中后，才将其触觉特征加入当前触觉特征均值。
4. 初始视觉特征与累积触觉特征融合，更新分类。
5. 达到置信度阈值、接触预算或候选耗尽后停止。

主动选择依赖候选的视觉特征，不使用类别标签或未选择候选的触觉特征作为选择依据。类别标签只用于组织及评估回放。

当前数据没有接触位置坐标或可执行的机器人动作。本流程只验证离线候选选择，不等同于真实空间中的主动接触策略。

### 仿真与 TACTO 适配

MuJoCo 中球体落到传感器平面，计算接触和法向力。适配脚本根据球面和平面的几何相交估计压入深度，再调用 TACTO `Renderer.render_from_depth` 输出光学触觉图像。

本项目保留 TACTO 的官方渲染实现与配置，只移除未使用的 PyBullet 导入，并将废弃的 `np.float` 改为 `float`。完整说明见 [vendor/tacto_renderer/README.md](vendor/tacto_renderer/README.md)。

它不包含 `tacto.Sensor`、PyBullet 集成、完整机械臂操作或校准后的弹性体形变模型。仿真检查和真实视频分类目前分别运行。

## 实验参数

默认配置位于 `configs/smoke.yaml`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `seed` | 42 | Python、NumPy、PyTorch 随机种子 |
| `image_size` | 64 | 模型输入边长 |
| `epochs` | 12 | 训练轮数 |
| `batch_size` | 24 | 批大小 |
| `learning_rate` | 0.003 | Adam 学习率 |
| `device` | `cpu` | 默认已验证训练设备 |
| `max_contacts` | 3 | 每个回放的接触预算 |
| `min_contacts` | 1 | 最少接触次数，确保验收时执行触觉更新 |
| `confidence_threshold` | 0.85 | 分类停止阈值 |

可以复制配置再修改，避免覆盖默认基线：

```bash
cp configs/smoke.yaml configs/custom.yaml
.venv/bin/python scripts/smoke_test.py --config configs/custom.yaml
```

数据抽帧数量可以单独调整：

```bash
.venv/bin/python scripts/prepare_data.py --frames-per-contact 3
```

配置应满足 `1 <= min_contacts <= max_contacts`。输出使用固定文件名，每次重跑会更新结果；需要保留某次结果时，请先备份 `outputs/`。

## 输出文件

| 路径 | 内容 |
| --- | --- |
| `data/download_manifest.json` | 下载版本、88 个文件的大小与 SHA256 |
| `data/processed/samples.json` | 图像对索引和来源信息 |
| `data/processed/summary.json` | 类别统计、划分和对齐误差 |
| `outputs/实验一测试报告.md` | 汇总数据、训练、接触和仿真的中文报告 |
| `outputs/smoke_report.json` | 训练配置、损失、分类指标和执行检查 |
| `outputs/simulation_report.json` | 接触力、压入深度和渲染差异 |
| `outputs/contact_traces.json` | 每个回放的候选选择、概率、接触次数与停止原因 |
| `outputs/smoke_model.pt` | 模型参数、类别名称和配置 |
| `outputs/training_loss.csv` | 逐 epoch 训练损失 |
| `outputs/smoke_results.png` | 训练损失与分类结果图 |
| `outputs/active_demo.gif` | 真实录像图像上的识别回放 |
| `outputs/data_preview.jpg` | 配对图像预览 |
| `outputs/mujoco_scene.png` | 仿真场景 RGB 图像 |
| `outputs/tacto_contact.png` / `tacto_no_contact.png` | 接触／未接触的触觉渲染 |
| `outputs/contact_indentation.npy` | 几何压入深度图 |
| `outputs/*.log` | 数据处理、训练和仿真日志 |

## 已有最小测试结果

以下为 2026-10-28、默认配置、CPU 上的实际执行记录，供复现时核对。结果文件不会随源码上传，需本地运行生成。

- 依赖检查通过；视频和时间戳加载、录像分离、梯度及参数更新、模型保存／重载一致性检查通过。
- 训练平均损失从约 **1.1003** 降至 **0.3966**。

| 输入 | 测试准确率 | 类别均衡准确率 |
| --- | ---: | ---: |
| 仅视觉 | 65.66% | 25.79% |
| 仅触觉 | 62.63% | 59.13% |
| 视触融合 | 64.65% | 25.40% |

测试集严重偏向 Rock，恒预测 Rock 即可获得 **84.85%** 准确率。因此这些结果只说明训练与评测能运行，不能说明分类性能已达标，或融合优于单模态。

随机和主动策略各运行 6 个测试录像，准确率均为 66.67%，平均接触 1 次。两者均在首次接触后达到阈值，没有证实主动选择具有优势；模型的高置信度也未经校准。

仿真检查执行 500 步，稳定法向力约 **0.981 N**，最大几何压入约 **0.601 mm**。接触／未接触触觉图像的平均像素绝对差约 **1.1946**，通过图像随接触变化的检查。

## 常见问题

### `Python 3.12 is required`

安装 Python 3.12，或通过 `PYTHON_BIN` 指定已有解释器。不要使用 macOS 自带的旧版本 Python。若从其他目录搬入了 `.venv`，推荐重新创建环境，避免旧激活脚本中的绝对路径失效。

### 镜像下载失败

先调低并发并重跑：

```bash
.venv/bin/python scripts/download_data.py --workers 2
```

确认当前网络可直连 `hf-mirror.com` 及下载重定向到的存储域名。脚本明确不使用代理，也不关闭证书校验。通过校验的文件会跳过。

### 图形测试报 `IndexError: list index out of range` 或找不到显示设备

在已登录图形桌面的本机终端运行 `scripts/simulation_smoke.py`。某些沙盒执行环境无法枚举 macOS 显示设备，需要允许本地图形访问。无显示会话的服务器需另行配置合适的 OpenGL 后端，本仓库未验证该部署方式。

### `MPS requested but unavailable`

保持默认 `device: cpu`。GPU 可用性会受操作系统和执行环境权限影响；单独通过 MPS 算子检查不代表完整训练流程已在该设备验证。

### SciPy 或旧渲染依赖导入失败

先使用完整锁定文件重装并检查：

```bash
bash scripts/setup.sh
.venv/bin/python -m pip check
```

已验证环境使用 SciPy 1.14.1。此前在本机测试时 SciPy 1.15.3 wheel 曾出现动态库加载问题。请先复现锁定环境，再逐项升级依赖。

### `import tacto` 失败

这是预期的包名区别：本项目提供的是 `vendor/tacto_renderer`，不是完整的 pip `tacto` 包。请直接运行 `scripts/simulation_smoke.py`，或参考其中的导入方式。

### 找不到 `outputs/实验一测试报告.md`

首次克隆不会带结果文件。先下载数据并运行 `bash scripts/run_smoke.sh`；报告需要数据准备、训练和仿真三个阶段全部完成。

## 后续实验方向

- 扩充真实数据和独立物体数量，建立更均衡的数据划分。
- 校准分类置信度，比较不同接触预算下的准确率和平均接触次数。
- 用空间接触坐标和可执行动作替换离线候选，实现闭环主动感知。
- 在更大测试集上比较随机策略、视觉差异策略与学习得到的策略。
- 对触觉几何和材料参数进行校准，再连接机械臂场景。

## 来源与第三方许可

详细数据版本和引用入口见 [references/SOURCES.md](references/SOURCES.md)。

- TACTO 渲染代码沿用上游 **MIT** 许可，原文保留于 [vendor/tacto_renderer/LICENSE](vendor/tacto_renderer/LICENSE)。
- 标签元数据来自 SeeingThroughTouch 仓库，保留其 **Apache 2.0** 许可于 [references/SeeingThroughTouch-LICENSE](references/SeeingThroughTouch-LICENSE)。
- Touch and Go 视频仍由其原作者及发布方管理；第三方镜像没有附带明确的 dataset card 或 license 元数据。本仓库不重新发布原始视频，也不对其授予新许可。
- 本仓库未为自编写代码另行指定整体开源许可，第三方许可不自动覆盖项目全部文件。
