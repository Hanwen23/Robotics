"""Write a readable Chinese report from actual smoke-test outputs."""
import json
from datetime import datetime
import platform
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'outputs'
    data = json.loads((ROOT / 'data/processed/summary.json').read_text())
    download = json.loads((ROOT / 'data/download_manifest.json').read_text())
    training = json.loads((output / 'smoke_report.json').read_text())
    simulation = json.loads((output / 'simulation_report.json').read_text())
    assert training['status'] == simulation['status'] == 'passed'
    rows = '\n'.join(f"| {name} | {metric['accuracy']:.2%} | {metric['balanced_accuracy']:.2%} |" for name, metric in training['classification'].items())
    replay = '\n'.join(f"| {name} | {metric['episodes']} | {metric['accuracy']:.2%} | {metric['mean_contacts']:.2f} |" for name, metric in training['contact_replay'].items())
    generated_date = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    replay_note = ('本次两种策略均在首次接触后达到阈值，指标相同。'
                   if all(x['mean_contacts'] == 1 for x in training['contact_replay'].values())
                   and len({x['accuracy'] for x in training['contact_replay'].values()}) == 1
                   else '本次接触次数与准确率以表中实际执行结果为准。')
    text = f'''# 实验一环境及最小测试记录

执行状态：通过。记录生成日期：{generated_date}。此结果只验证可运行性。

## 环境

执行平台：{platform.system()} / {platform.machine()}，项目内 `.venv`，Python {platform.python_version()}。
PyTorch {training['torch_version']}，{simulation['physics']}，TACTO 0.0.3 的独立渲染器。
分类测试使用 {training['device']}，无需下载预训练权重。完整依赖见 `requirements.lock.txt`。
TACTO 适配的来源、许可和两处代码修改见 `vendor/tacto_renderer/README.md`。

## 下载和数据检查

- 实际渠道：HF Mirror 直连，curl 强制禁用代理；Python 包使用清华镜像。
- 数据仓库：`{download['repo']}`，固定版本 `{download['revision']}`。
- 下载 {download['sequences']} 段录像、{len(download['files'])} 个文件、{download['total_bytes'] / 1e6:.2f} MB。
- 所有文件通过大小与 SHA256 核验。清单位于 `data/download_manifest.json`。
- 这是社区提供的公开 probe 子集，**不是 Touch and Go 完整数据集**。
- 接触区间、材质标签来自 SeeingThroughTouch 的公开元数据，来源版本见 `references/SOURCES.md`。
- 有标签匹配的录像 {data['matched_labeled_sequences']} 段。分类使用 {data['used_sequences']} 段录像、{data['contacts']} 个接触区间、{data['pairs']} 对图像。
- 分类类别：{', '.join(data['classes'])}。Grass 只有一段录像，未纳入分类划分；其原始数据已下载。
- 每个接触区间抽取 3 帧，触觉帧按最近时间戳匹配。最大误差 {data['max_alignment_error_ms']:.2f} ms。
- 按录像划分：训练 {training['train_pairs']} 对，测试 {training['test_pairs']} 对，录像交集为空。
- 这是本地自定义划分；不同录像仍可能包含相同物理对象，不是官方基准评测。

## 真实数据分类测试

双分支小型 CNN，分别输出仅视觉、仅触觉和拼接融合分类结果。
采用类别均衡采样和固定随机种子，训练 {training['config']['epochs']} 个 epoch。
平均训练损失从 {training['history'][0]['loss']:.4f} 降至 {training['history'][-1]['loss']:.4f}。
已验证梯度有限、优化器更新参数，以及权重保存／重新加载的一致性。

| 输入 | 测试准确率 | 类别均衡准确率 |
| --- | --- | --- |
{rows}

测试集严重不均衡，恒预测 Rock 的准确率为 {max(data['class_counts']['test'].values()) / training['test_pairs']:.2%}。
因此上表不能说明模型达到可用识别性能，也不能说明融合优于单模态。

## 主动接触最小回放

同一测试录像中的已记录接触作为候选，主动策略根据可见视觉特征的差异选择下一个候选，
选中后才使用该候选的触觉特征更新分类。随机策略在相同候选池上运行。
停止条件：置信度达到 {training['config']['confidence_threshold']}、接触满 {training['config']['max_contacts']} 次或候选耗尽。
为确保最小测试实际执行触觉分支，配置要求至少接触 {training['config']['min_contacts']} 次。

| 策略 | 测试录像数 | 准确率 | 平均接触次数 |
| --- | --- | --- | --- |
{replay}

{replay_note}候选选择记录见 `contact_traces.json`。
数据没有接触空间坐标，当前实现是离线接触候选回放，不是在线机器人选点，也没有证实主动策略更好。
该样本上的高置信度不代表经过校准的正确率。

## 物理接触与触觉光学渲染

MuJoCo 执行 {simulation['physics_steps']} 步，球体落在传感器平面，产生接触。
稳定法向力 {simulation['settled_normal_force_n']:.4f} N，最大几何压入深度 {simulation['maximum_indentation_m'] * 1000:.3f} mm。
将球面和平面的几何压入图送入 TACTO 官方光学渲染器，输出 {simulation['rgb_shape']} 触觉图像。
接触与未接触图像的平均像素差为 {simulation['contact_vs_no_contact_rgb_mean_absolute_difference']:.4f}，已验证大于零。
MuJoCo 场景 RGB 渲染也已通过。

此适配只连接球面／平面几何深度和光学渲染，未模拟真实弹性体形变，也不包含 PyBullet Sensor API。
它与真实视频分类测试分别验证仿真和数据两条路径，尚未组成机器人闭环控制系统。

## 输出和复现

- `smoke_report.json`：训练和接触回放机器可读结果。
- `simulation_report.json`：仿真和渲染测试结果。
- `smoke_model.pt`：本次训练的模型权重。
- `smoke_results.png`、`training_loss.csv`：真实运行结果图表。
- `active_demo.gif`：真实数据上的最小识别回放，包含一次触觉更新。
- `data_preview.jpg`、`mujoco_scene.png`、`tacto_contact.png`、`tacto_no_contact.png`：数据及渲染预览。

在项目根目录执行 `bash scripts/run_smoke.sh` 可离线重跑。图形渲染需要能访问 macOS 显示会话的终端；
Codex 沙盒内看不到显示设备时，仿真脚本需要在允许本地图形访问的执行环境运行。
'''
    (output / '实验一测试报告.md').write_text(text)
    print(output / '实验一测试报告.md')


if __name__ == '__main__':
    main()
