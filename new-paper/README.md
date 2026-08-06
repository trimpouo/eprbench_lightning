# new-paper/ — 新型论文研究包（Agent-Readable Research Package）

本文件夹是本项目的**“新型论文”层**：它不取代 `docs/` 中的叙事文档，而是把整个研究整理成一个
**同时供人、LLM 和研究智能体读取、检验、运行与继续扩展的可执行研究包**。

设计理念来自 Andrej Karpathy 近年公开讨论中拼合出的一种学术成果形态：

- **LLM-friendly 文档**：内容以单一、清晰、机器可读的 Markdown/YAML/TSV 组织，可直接进入上下文窗口；
- **LLM Wiki**：原始资料（`sources/`）与综合判断（`paper.md`、`evidence/`）分层保存，综合层可持续更新；
- **autoresearch**：实验以结构化记录（`experiments/results.tsv`）保存，智能体可读取目标、复现实验、按统一指标保留或放弃改动（`agent.md`）。

## 目录结构

```
new-paper/
├── README.md            # 本文件：理念与使用说明
├── paper.md             # 论文入口：机器可读元数据 + 主张清单 + 正式叙事
├── claims.yaml          # 核心主张 C1–C5（机器可读）
├── agent.md             # 给研究智能体的操作说明（复现 / 审查 / 继续研究）
├── evidence/
│   ├── evidence-map.md  # 主张 → 实验 → 数据的映射，含反例与统计状态
│   ├── failure-cases.md # 失败实验与负面结果（不进入主线但保留）
│   └── mlflow-analysis.md # logs/mlflow.db 全量实验数据库判读（116 runs）
├── experiments/
│   ├── README.md        # 每张实验记录的复现命令
│   ├── results.tsv      # 全部实验的结构化记录（R@1/5/10/100）
│   └── mlflow_runs.tsv  # logs/mlflow.db 的 run 级全量导出
├── sources/
│   ├── bibliography.bib # 引用文献
│   └── source-notes/    # 单篇文献的结构化阅读记录（持续补充）
├── CHANGELOG.md         # 研究包版本历史
└── CITATION.cff         # 引用元数据
```

## 与本仓库已有内容的对应关系

研究包**不复制**已有成果，只做索引与结构化：

| 研究包文件 | 对应的仓库现有内容 |
|---|---|
| `experiments/results.tsv` | `docs/eventvpr_exploration_journey.md`、`docs/nyc_vpr_sacf_scfm_ablation.md`、`README.md` 中的结果表 |
| `experiments/README.md` 中的复现命令 | `configs/experiment/*.yaml`、`scripts/*.sh`、`AGENTS.md` 中的训练/评估 recipe |
| `evidence/failure-cases.md` | `docs/eventvpr_exploration_journey.md` 附录 A.5 及各文档的“踩坑”记录 |
| `sources/bibliography.bib` | `README.md` Citation、EPRBench 论文、NYC-Event-VPR、DINOv2、CricaVPR、CLIP |
| 代码与环境 | `src/`、`configs/`、`requirements.txt`、`environment.yaml`（研究包不另存副本） |

## 五条维护原则

1. **机器可读**：新结论必须同步进 `claims.yaml` 与 `results.tsv`，不允许只写在叙事文档里。
2. **可执行**：`results.tsv` 中每一行都必须能在 `experiments/README.md` 找到复现命令。
3. **可追踪**：`paper.md` 中的每个主张必须编号（Cx），并链接到实验编号（Ex）。
4. **可更新**：新证据出现时更新对应主张的置信度与边界，而不是整篇重写。
5. **可继续研究**：任何智能体/新协作者应先读 `agent.md`，再动手。
