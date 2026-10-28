# 数据和代码来源

## 课程范围

实验范围依据用户提供的《课程实验选题说明》第 3–8 页：视触分类、主动探索及随机策略对照。
课件不属于运行依赖，未收入源码仓库。

## Touch and Go

- [原始项目](https://touch-and-go.github.io/)
- [原始论文](https://arxiv.org/abs/2211.12498)
- [原始作者代码](https://github.com/fredfyyang/Touch-and-Go)
- [实际使用的 HF Mirror 下载入口](https://hf-mirror.com/datasets/kingJulio/touch-and-go-probe)
- [镜像的 Hugging Face 仓库](https://huggingface.co/datasets/kingJulio/touch-and-go-probe)
- 固定版本：`ba0bee2d863aabee6b3d533a4e15618d3d149ab1`。

该镜像由第三方维护，包含 22 段录像，是原始数据集的子集。它未提供 dataset card 或 license 元数据，本仓库不授予视频数据新许可，也不重新发布原始视频。

文件索引及上游 SHA256 保存在 `hf_dataset_tree.json`。本次下载的文件路径、大小及 SHA256 保存在 `../data/download_manifest.json`；文件路径相对项目根目录，不包含本机绝对路径。

原始项目将 `video.mp4` 定义为视觉视频，`gelsight.mp4` 定义为触觉视频，`time1.npy` 和 `time2.npy` 分别记录其时间戳。本项目使用最近时间戳配对，并保存对齐误差。

## 接触区间和材质标签

- [SeeingThroughTouch 仓库](https://github.com/kaistmm/SeeingThroughTouch)
- 固定版本：`edd0e04430b966fcf08b8e17fe9955f71a2dcbed`。
- 原始目录：`datasets/touch_and_go/metadata/`。
- [类别映射文件](https://github.com/kaistmm/SeeingThroughTouch/blob/edd0e04430b966fcf08b8e17fe9955f71a2dcbed/datasets/touch_and_go/metadata/category_reference.txt)
- [训练接触区间文件](https://github.com/kaistmm/SeeingThroughTouch/blob/edd0e04430b966fcf08b8e17fe9955f71a2dcbed/datasets/touch_and_go/metadata/train_nointer_touch_instances.txt)
- [用于核验格式的上游数据加载代码](https://github.com/kaistmm/SeeingThroughTouch/blob/edd0e04430b966fcf08b8e17fe9955f71a2dcbed/dataset.py)

本仓库保留原样的 `category_reference.txt` 和 `train_nointer_touch_instances.txt`，校验值见 `source_checksums.json`。没有复制上游完整训练代码或与本次划分无关的测试标签。

上游仓库提供 Apache License 2.0，原文保留于 `SeeingThroughTouch-LICENSE`。本项目对上游 train 元数据重新按录像划分，因此不能称为复现其官方评测。

## TACTO

- [官方项目](https://github.com/facebookresearch/tacto)
- [官方 PyPI 发行包](https://pypi.org/project/tacto/0.0.3/)
- 使用 `tacto==0.0.3` wheel 中的 `renderer.py` 和 `config_digit.yml`，通过清华 PyPI 镜像获取。
- 上游 wheel SHA256：`dd62f0e4b75374062d5795e5ea18a5d107908d8141e272a2fcde1a4f60d44ddf`。
- 原始许可为 MIT，保留于 `../vendor/tacto_renderer/LICENSE`。
- 修改与适配范围见 `../vendor/tacto_renderer/README.md`。

wheel 仅为本地来源核验材料，不收入仓库；需要运行的渲染源码、配置和许可已收入 `vendor/`。

## MuJoCo

- [官方项目](https://github.com/google-deepmind/mujoco)
- [官方文档](https://mujoco.readthedocs.io/)

使用 `mujoco==3.3.7` 的 PyPI wheel，经清华镜像安装，不在本仓库重新打包 MuJoCo。

## 网络设置

视频下载脚本通过 `curl --noproxy '*'` 强制直连并保留 TLS 验证。安装脚本清除本次进程的代理变量，使用 `https://pypi.tuna.tsinghua.edu.cn/simple`。

必要元数据已随源码提供。只有手动删除这些标签文件时，下载脚本才需要访问固定版本的 GitHub Raw 地址补齐。脚本不修改系统代理、全局 pip 配置或系统 Python。
