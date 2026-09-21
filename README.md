# HYworld2Atlas

Replace HY-WorldPlay's FOV-based history retrieval with a **feed-forward 4D Gaussian asset** — turning the interactive world model's memory into a persistent 3D/4D scene that can be observed from any viewpoint (Atlas-style) and queried by camera pose instead of chunk retrieval.

English below.

---

## 项目目标

原始问题（`idea.txt`）：HunyuanWorldplay 的历史召回模块用相机 FOV 检索相近 chunk，视频变长后**检索耗时超过生成耗时**。本项目用 3D 资产维护历史，召回时给定位姿直接渲染该视角的内容。

```
┌────── GPU A · HY 环境 ──────┐         ┌────── GPU B · MoVieS 环境 ──────┐
│  HY DiT (t 阶段去噪)          │ ──更新─► │  VAE decode → MoVieS 编码        │
│                              │         │  → 全局 4D 高斯资产 (t-1 更新)   │
│  召回: 按 (t', 相机) 查询     │ ◄─渲染─ │  gsplat 透视渲染任意视角          │
│  t-2 版本的资产               │         │                                 │
└──────────────────────────────┘         └─────────────────────────────────┘
```

## 仓库结构

| 目录 | 说明 |
|---|---|
| `gs_bridge/` | **本项目主体**：HY ↔ MoVieS 桥接层（设计文档、双端代码、UI、测试） |
| `HY-WorldPlay/` | 腾讯混元 HY-World 1.5 交互式世界模型（第三方，未改动） |
| `MoVieS/` | 前馈 4D 动态高斯（第三方，未改动） |
| `StreamSplat/` | 已弃用方案的参考（第三方，未改动） |
| `idea.txt` | 研究构想原文 |

主体文档入口：`gs_bridge/README.md`（目的、操作说明、进度、测试声明，中英双语）与 `gs_bridge/DESIGN.md`（设计定稿 + 决策记录）。

## ⚠️ 当前状态

**除 M0 的 21 个纯数学单元测试外，完全没有开始测试**——MoVieS 编码、gsplat 渲染、HY 接入、IPC、UI 均未运行过。详见 `gs_bridge/README.md` 的测试声明。

---

# HYworld2Atlas (English)

Replace HY-WorldPlay's FOV-based history retrieval with a **feed-forward 4D Gaussian asset** — turning the interactive world model's memory into a persistent 3D/4D scene that can be observed from any viewpoint (Atlas-style) and queried by camera pose instead of chunk retrieval.

## Goal

The original problem (`idea.txt`): HunyuanWorldplay's history recall retrieves similar chunks by camera FOV; as videos grow long, **retrieval time exceeds generation time**. This project maintains history as a 3D asset — a recall renders the requested view directly from the asset at the given pose.

## Repository Layout

| Directory | Description |
|---|---|
| `gs_bridge/` | **The project proper**: HY ↔ MoVieS bridge layer (design doc, both-side code, UI, tests) |
| `HY-WorldPlay/` | Tencent Hunyuan HY-World 1.5 interactive world model (third-party, unmodified) |
| `MoVieS/` | Feed-forward 4D dynamic Gaussians (third-party, unmodified) |
| `StreamSplat/` | Reference from a dropped approach (third-party, unmodified) |
| `idea.txt` | Original research idea (Chinese) |

Main documentation: `gs_bridge/README.md` (purpose, operations, status, testing disclaimer — bilingual) and `gs_bridge/DESIGN.md` (design decisions).

## ⚠️ Current Status

**Apart from M0's 21 pure-math unit tests, nothing has been tested** — MoVieS encoding, gsplat rendering, HY integration, IPC, and the UI have never been run. See the testing disclaimer in `gs_bridge/README.md`.
