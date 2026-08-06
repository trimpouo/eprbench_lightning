# Source Note — EPRBench (Wang et al., 2026)

- **Bibkey**: `wang2026eprbench`
- **原文**: https://arxiv.org/abs/2602.12919
- **本仓库存档**: `docs/EPRBench A high-quality benchmark dataset for event stream based visual place recognition.md`
- **原始资料状态**: 只读，不修改；本笔记是综合层，可更新。

## 与本研究包的关系

- 本仓库是其 **Lightning + Hydra 复现**，复现对象是其中的 SGVPR 方法。
- 官方权重预期指标（EventVPR test）：R@1/5/10/100 = 94.0/96.0/96.9/99.2。
  本研究包 E01（93.6）与之存在 0.4 个百分点的差距，原因未完全定位，
  已知差异包括 `network.py` 中 masked fill 由 -1e9 改为 -1e4（fp16 数值妥协，
  见根目录 README 的 Implementation note）。

## 关键内容摘录（综合）

- 数据集：10K event 序列 / 65K event 帧，手持 + 车载采集，多视角/天气/光照。
- 提供 LLM 生成 + 人工修正的场景描述（`scene_descriptions/`）。
- SGVPR 范式：文本描述引导空间注意力 token 选择、跨模态融合与多尺度表征学习。

## 开放疑问

- 官方训练是否 fp32？（本复现默认 16-mixed，可能是 E01 差距来源之一。）
- 官方在多数据集上的泛化结果缺失，C2 的跨数据集结论为本文档独有贡献。
