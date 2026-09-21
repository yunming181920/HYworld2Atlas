# gs_bridge — 4D Gaussian Asset Memory for HY-WorldPlay

**中文** | [English](#english)

---

## 目的 / Purpose

用**前馈 4D 高斯资产**替换 HY-WorldPlay（HunyuanWorldplay）的 FOV 历史召回模块。

原始问题（`../idea.txt`）：HY-WorldPlay 的历史召回用相机 FOV 检索相近 chunk，视频变长后**检索时间超过生成时间**。本项目改为用 3D 资产维护历史——召回时给定位姿，从资产直接渲染该视角的内容。

核心架构（双工双卡管道）：

```
┌────── GPU A · HY 环境 ──────┐         ┌────── GPU B · MoVieS 环境 ──────┐
│  HY DiT (t 阶段去噪)          │ ──更新─► │  VAE decode → MoVieS 编码        │
│                              │         │  → 全局 4D 高斯资产 (t-1 更新)   │
│  召回: 按 (t', 相机) 查询     │ ◄─渲染─ │  gsplat 透视渲染任意视角          │
│  t-2 版本的资产               │         │                                 │
└──────────────────────────────┘         └─────────────────────────────────┘
```

- **流式资产**：每个生成完的 chunk 经 MoVieS 前馈编码成一个高斯窗口，并入固定容量环形缓冲（FIFO，容量 20，对标 Helios 的固定容量上下文压缩）
- **视角召回**：查询 = 给定位姿从资产渲染，替代"FOV 检索 chunk 索引 + 拷贝历史 latent"
- **t-2 滞后**：去噪 chunk t 时只保证 chunk ≤ t-2 已入资产（DiT 刚完成 t-1，其编码还在路上）——按 idea.txt 的双工时序显式建模
- **Atlas 式观察**：时间静止（暂停）时从任意相机视角观察资产；换视角继续生成 = 资产回滚到 t-2 + 新视角渲染参考帧 + 硬切重开 rollout（无缝过渡不经训练不可行，已明确不做）

高斯方案选型沿革：StreamSplat（正交相机，弃）→ MVSplat（透视静态，弃）→ **MoVieS**（透视 + 4D 动态，现方案）。详见 `../CLAUDE.md`。

## 支持的操作 / Supported Operations

### 键盘操作（与 HY-WorldPlay 原生操作同构）

| 键 | 动作 | 步长 |
|---|---|---|
| W / S | 前进 / 后退 | 0.08 单位/步 |
| A / D | 左移 / 右移 | 0.08 单位/步 |
| ← / → | 视角左转 / 右转（yaw） | 3°/步 |
| ↑ / ↓ | 视角上仰 / 下俯（pitch） | 3°/步 |
| Space | 暂停 / 继续 | |
| Esc | 退出 | |

一个运动步 = 1 个 latent（与 HY 轨迹串 `w-31` 同粒度）。两种状态下同一套键位：

- **生成中**：按键录入待生成队列，driver 在 chunk 边界展开成轨迹段续给 rollout（HY 原有的轨迹串/pose JSON 输入方式同样保留）
- **暂停后**：按键驱动观察相机在 4D 资产内自由飞行（实时渲染回传）

### UI 按钮

| 按钮 | 语义 |
|---|---|
| ⏸ 暂停 | 在 chunk 边界挂起生成，主视图切换为资产观察（冻结在最新完成版本） |
| ▶ 继续 | 相机未动 → 原轨迹无损续播；相机已动 → 资产回滚 t-2 + 当前视角渲染参考帧 + 从新视角硬切重开生成 |
| ⟲ 复原 | 观察相机平滑滑回暂停时刻的生成位姿（复原后"继续"即原轨迹续播） |

## 当前进度 / Current Status

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 帧↔latent↔chunk 时间映射、HY↔MoVieS 相机转换、SE3 位姿插值、canonical 归一化 + 21 个单元测试 | ✅ 完成（测试已通过） |
| M1 | MoVieS encoder 封装（绕 VGGT 下载/resources 断言坑）+ 编码质量探针脚本 | ✅ 代码完成 |
| M2 | GlobalAsset 环形缓冲（FIFO/版本门/回滚）+ (t, 相机) 视角查询 | ✅ 代码完成 |
| M3 | Atlas 冻结时间轨道相机预览脚本 | ✅ 代码完成 |
| M4 | 双进程 IPC（TCP + JSON 头协议）：TRAJECTORY_SYNC / CHUNK_UPDATE / QUERY / ATLAS | ✅ 代码完成 |
| M4.5 | 换视角重生成：资产回滚、VIEWPOINT_RESET、在途更新防串扰（双重守卫） | ✅ 代码完成 |
| M5 | HY 端接入：rollout fork（两个标记块）、记忆 latent 替换、VAE 编码对齐 | ✅ 代码完成 |
| M7 | pygame UI 播放器：帧展示 + 三按钮 + WASD 双状态操作 + ZMQ 双进程 | ✅ 代码完成 |
| M6 | 端到端对比实验（原 FOV 召回 vs 资产召回：质量 + 长视频耗时曲线） | ⬜ 未开始 |

```
gs_bridge/
├── DESIGN.md            # 设计定稿（v4 + 换视角语义，含决策记录）
├── common/              # 双端共用：time_map / camera_conv / types / ipc_protocol
├── asset_server/        # GPU B：encoder / asset / query / server
├── hy_client/           # GPU A：client / latent_prep / rollout_fork / memory_replace
├── offline/             # GPU 就绪后的一次性验证脚本（探针/资产自检/Atlas 预览）
├── ui/                  # M7：keys / camera / player / driver / run_ui
└── tests/               # M0 单元测试
```

## ⚠️ 测试声明 / Testing Disclaimer

**除 M0 的 21 个纯数学单元测试（相机转换、时间映射、SE3 插值）外，本项目完全没有开始测试。**

具体而言，以下均**未运行过**：

- MoVieS 模型加载与编码（GPU B 路径）——VGGT-1B 下载、checkpoint 加载、`SplatRecon` 构建均未验证
- gsplat 渲染（所有查询/观察/参考帧渲染路径）
- HY pipeline 接入（rollout fork 与真实 pipeline 的 API 对齐、VAE encode 数值对齐）
- 双进程 IPC（协议往返、版本门竞态、换视角回滚竞态）
- UI（pygame 窗口运行、ZMQ 推流、暂停/继续/复原交互）

M1–M5、M7 的所有代码只做过语法检查（AST parse）和 headless 逻辑验证（键位常量、相机数学、continue 分支判定），**没有任何一次真实的端到端运行**。预期在 GPU 环境就绪后按 `offline/` 脚本顺序验证（encode_probe → asset_test → atlas_preview → 端到端）。已知风险点（VAE 因果前缀对齐、224 编码 vs 832×480 宽高比、驱动接缝）记录在各文件注释与 DESIGN.md 中。

---

<a name="english"></a>

# gs_bridge — 4D Gaussian Asset Memory for HY-WorldPlay

[中文](#目的--purpose) | **English**

---

## Purpose

Replace HY-WorldPlay's (HunyuanWorldplay) FOV-based history retrieval with a **feed-forward 4D Gaussian asset**.

The original problem (`../idea.txt`): HY-WorldPlay's history recall retrieves similar chunks by camera FOV; as videos grow long, **retrieval time exceeds generation time**. This project maintains history as a 3D asset instead — a recall is "given a pose, render that view from the asset".

Core architecture (duplex dual-GPU pipeline):

```
┌────── GPU A · HY env ────────┐         ┌────── GPU B · MoVieS env ────────┐
│  HY DiT (denoising chunk t)  │ -update► │  VAE decode → MoVieS encode       │
│                              │         │  → global 4D Gaussian asset (t-1) │
│  recall: query the asset     │ ◄render─ │  gsplat perspective rendering     │
│  at version t-2              │         │  from any viewpoint               │
└──────────────────────────────┘         └───────────────────────────────────┘
```

- **Streaming asset**: every finished chunk is feed-forward encoded into a MoVieS Gaussian window and appended to a fixed-capacity ring buffer (FIFO, capacity 20 — the Helios-style bounded context compression)
- **Viewpoint recall**: a query renders from the asset at a given pose, replacing "FOV-retrieve chunk indices + copy history latents"
- **t-2 lag**: while chunk t denoises, only chunks ≤ t-2 are guaranteed in the asset (the DiT just finished t-1; its encoding is still in flight) — modeled explicitly per idea.txt's duplex timing
- **Atlas-style observation**: with time frozen (paused), observe the asset from any camera; continuing from a new viewpoint = roll the asset back to t-2 + render the reference frame at the new pose + hard-cut restart of the rollout (seamless continuation without training is impossible and explicitly out of scope)

Gaussian backend history: StreamSplat (orthographic camera, dropped) → MVSplat (perspective, static, dropped) → **MoVieS** (perspective + 4D dynamic, current). Details in `../CLAUDE.md`.

## Supported Operations

### Keyboard (isomorphic with HY-WorldPlay's native controls)

| Key | Action | Step |
|---|---|---|
| W / S | forward / backward | 0.08 units/step |
| A / D | strafe left / right | 0.08 units/step |
| ← / → | yaw left / right | 3°/step |
| ↑ / ↓ | pitch up / down | 3°/step |
| Space | pause / continue | |
| Esc | quit | |

One motion step = 1 latent (same grain as HY's pose string `w-31`). The same keymap serves both states:

- **While generating**: keystrokes enqueue motions; the driver extends the trajectory at chunk boundaries (HY's original pose-string / pose-JSON input remains supported)
- **While paused**: the same keys fly the observation camera freely through the 4D asset (live renders)

### UI buttons

| Button | Semantics |
|---|---|
| ⏸ Pause | Suspend generation at a chunk boundary; main view switches to asset observation (frozen at the latest complete version) |
| ▶ Continue | Camera unmoved → resume the original rollout losslessly; camera moved → asset rollback to t-2 + reference frame rendered at the current pose + hard-cut regeneration from the new viewpoint |
| ⟲ Restore | Smoothly slerp the observation camera back to the pause-time generation pose ("continue" then degrades to plain resumption) |

## Current Status

| Milestone | Scope | Status |
|---|---|---|
| M0 | frame↔latent↔chunk time mapping, HY↔MoVieS camera conversion, SE3 pose interpolation, canonical normalization + 21 unit tests | ✅ done (tests pass) |
| M1 | MoVieS encoder wrapper (VGGT-download / resources-assert workarounds) + encode-quality probe | ✅ code complete |
| M2 | GlobalAsset ring buffer (FIFO / version caps / rollback) + (t, camera) viewpoint queries | ✅ code complete |
| M3 | Atlas frozen-time orbit preview | ✅ code complete |
| M4 | Two-process IPC (TCP + JSON-header protocol): TRAJECTORY_SYNC / CHUNK_UPDATE / QUERY / ATLAS | ✅ code complete |
| M4.5 | Viewpoint-reset regeneration: asset rollback, VIEWPOINT_RESET, in-flight update guards (double protection) | ✅ code complete |
| M5 | HY-side integration: rollout fork (two marked blocks), memory-latent swap, VAE encode alignment | ✅ code complete |
| M7 | pygame player UI: frame view + three buttons + dual-state WASD + ZMQ two-process layout | ✅ code complete |
| M6 | End-to-end comparison (FOV recall vs asset recall: quality + long-video latency curves) | ⬜ not started |

```
gs_bridge/
├── DESIGN.md            # design doc (v4 + viewpoint-reset semantics, decision log)
├── common/              # shared: time_map / camera_conv / types / ipc_protocol
├── asset_server/        # GPU B: encoder / asset / query / server
├── hy_client/           # GPU A: client / latent_prep / rollout_fork / memory_replace
├── offline/             # one-shot validation scripts (probe / asset self-check / atlas preview)
├── ui/                  # M7: keys / camera / player / driver / run_ui
└── tests/               # M0 unit tests
```

## ⚠️ Testing Disclaimer

**Apart from M0's 21 pure-math unit tests (camera conversion, time mapping, SE3 interpolation), this project has NOT been tested at all.**

In particular, none of the following has ever been run:

- MoVieS model loading and encoding (GPU B path) — VGGT-1B download, checkpoint loading, `SplatRecon` construction are all unverified
- gsplat rendering (every query / observation / reference-frame path)
- HY pipeline integration (rollout-fork API alignment against the real pipeline, VAE-encode numerical alignment)
- Two-process IPC (protocol round-trips, version-cap races, viewpoint-reset rollback races)
- UI (pygame window run, ZMQ streaming, pause/continue/restore interaction)

All M1–M5/M7 code has only had syntax checks (AST parse) and headless logic verification (keymap constants, camera math, continue-branch decisions) — **not a single real end-to-end run**. Validation is planned to follow the `offline/` scripts once a GPU environment is available (encode_probe → asset_test → atlas_preview → end-to-end). Known risks (causal-VAE prefix alignment, 224 encode vs 832×480 aspect, driver seams) are documented in file comments and DESIGN.md.
