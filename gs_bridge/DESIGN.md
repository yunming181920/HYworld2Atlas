# gs_bridge 设计文档（v4 定稿）

HY-WorldPlay ↔ MoVieS 桥接层：用 MoVieS 前馈 4D 高斯维护流式 3D 资产，替换 HY 的 FOV 历史召回。

> 依据：`idea.txt` 四点构想 + CLAUDE.md 技术约束。本文档是实现的唯一设计基准。

## 0. 架构总览

```
┌────── GPU A · HY 环境 · 进程1 ──────┐         ┌────── GPU B · MoVieS 环境 · 进程2（常驻）──────┐
│ HY pipeline（改造点最小化）           │         │                                               │
│  chunk t-1 生成完 → VAE decode ─────┼──更新──►│  窗口编码（MoVieS backbone）                    │
│                                    │         │   → 记录归一化反变换 → 并入资产环形缓冲          │
│  chunk t 去噪前：                    │         │                                               │
│    选召回视角（预计算轨迹切片）────────┼──查询──►│  (t', 相机) → 选窗口 → 烘焙时间                  │
│    渲染RGB → VAE encode ─ latent    │◄────────┼──  → gsplat 渲染 → RGB/深度                     │
│    → 替换记忆帧来源，进 DiT          │         │  Atlas流：快照 + 任意相机（不阻塞更新）          │
└────────────────────────────────────┘         └───────────────────────────────────────────────┘
```

核心设计判断：

- **轨迹全程预计算**：`'w-31'` 在启动时展开为 32 个 latent 位姿（`pose_to_input`，generate.py:173）。更新流（chunk t-1 位姿）和查询流（chunk t 召回视角）都是这份数组的切片，无运行时位姿推导。
- **窗口对齐不需要估计**：MoVieS 输入是 canonical 归一化的（首相机→单位阵 + 尺度归一）。编码前我们自己做归一化、把反变换存进窗口——窗口→HY 全局系是已知解析反变换。
- **资产 = 窗口环形缓冲**：每 chunk 一个 MoVieS 窗口（model_outputs + 运动偏移 + 反变换），固定容量 FIFO 淘汰（对标 Helios / 原 memory_frames=20）。高斯融合/去重列第二阶段。

## 1. 帧位姿插值（v4 定稿，M0 已单测锁定）

**轨迹语义（M0 实现时核实修正）**：`'w-31'` = 31 个 motion = 32 个 latent 位姿，**相邻 latent 位姿间距 = 1 个运动步（0.08/3°），该步被 4 个像素帧分摊**——帧 f 的相机进度 = f/4 步（帧 124 = 第 31 步 = latent 31）。不是"每 4 帧共享 1 个位姿"，而是"1 步运动摊到 4 帧上"。

权威映射（`time_map.py`，21 个单测锁死）：

```
帧 f (0-indexed, 共 4L-3 帧, L = latent 数):
  f == 0  → latent 0（特例：latent 0 只覆盖帧 0）
  f >= 1  → latent (f+3)//4
chunk 0 = 帧 [0,13) 共13帧;  chunk c≥1 = 帧 [16c-3, 16c+13) 共16帧
```

**像素帧位姿 = 相邻 latent 位姿的 SE3 插值**（`camera_conv.interpolate_frame_poses`）：

- 帧属于 latent l 时：锚点 = latent l-1、latent l，`t = (f - 4(l-1)) / 4 ∈ [1/4, 1]`
- 平移线性 + 旋转 slerp（`_slerp_quat`），同 t 同步
- 帧 4l（latent 末帧）恰好等于 latent l 的位姿（t=1 精确命中，单测覆盖）
- 纯匀速轨迹下插值零误差；窗口内相机连续运动，立体匹配友好
- `AssetQuery` 传 **latent 索引**；渲染需 latent 内特定帧位姿时用同一插值函数

## 2. 目录结构（工作区根新建，不动两个仓库）

```
gs_bridge/
├── DESIGN.md                  # 本文档
├── config.py                  # Config：路径、分辨率、chunk参数、容量、IPC
├── common/                    # 双端共用，纯 torch/numpy，无重依赖
│   ├── types.py               # CameraPose / ChunkPacket / AssetQuery / AssetRender
│   ├── camera_conv.py         # HY↔MoVieS相机转换 + canonical归一化/反变换 + SE3插值
│   ├── time_map.py            # 帧↔latent↔chunk↔全局时间映射（含首帧特例）
│   └── ipc_protocol.py        # 消息定义与序列化
├── offline/                   # M1–M3 单进程离线验证（MoVieS 环境）
│   ├── encode_probe.py        # HY视频+位姿→编码→已知视角回投，量化PSNR
│   ├── asset_test.py          # 资产缓冲+查询接口单测（延迟/显存）
│   └── atlas_preview.py       # 冻结时间轨道相机mp4
├── asset_server/              # GPU B 核心
│   ├── encoder.py             # MoVieS backbone 封装（处理VGGT下载坑、resources断言坑）
│   ├── asset.py               # GlobalAsset：环形窗口缓冲、版本管理、FIFO淘汰
│   ├── query.py               # (t',相机)→选窗口→烘焙时间→gsplat渲染
│   ├── server.py              # 进程入口：IPC循环
│   └── atlas.py               # Atlas观察器（快照渲染，双缓冲）
└── hy_client/                 # GPU A 侧（薄）
    ├── client.py              # 发chunk / 查资产（带版本上限）
    ├── memory_replace.py      # 替换_ar_rollout_inner记忆帧来源（monkey-patch注入）
    └── latent_prep.py         # 渲染RGB→HY VAE encode→context latent
```

## 3. 核心数据流

```python
# common/types.py（示意）
CameraPose    # c2w(4,4) + 归一化fxfycxcy(4) + frame_idx + chunk_idx
ChunkPacket   # frames(F,3,H,W) + chunk_idx        # A→B 更新包（位姿server已存，按chunk_idx取）
AssetQuery    # version_cap(=t-2) + pose_indices(latent索引) + render_size
AssetRender   # images(V,3,H,W) + depths(V,1,H,W) + version   # B→A 应答
TrajectorySync  # 启动时一次性：全轨迹HY→MoVieS转换结果（含插值表）
```

- **TrajectorySync**（启动一次）：全轨迹 32+ latent 位姿 → MoVieS c2w/fxfycxcy 逐帧插值表 → server 缓存。之后 ChunkPacket 只传帧像素，AssetQuery 只传 latent 索引。
- **更新流**（每 chunk）：DiT 完成 chunk t-1 → VAE decode 16 帧 → IPC → server：resize 224 → **canonical 归一化（首帧位姿→单位阵 + 尺度，反变换存进窗口）** → `backbone(images, C2W, fxfycxcy, input_t, output_t)` → WindowData 入缓冲 → version=t-1
- **查询流**（每 chunk 生成前）：按 `query_pose_mode` 选召回 latent 索引 → `AssetQuery(version_cap=t-2)` → server 定位 (窗口, output_t) → 烘焙运动偏移 → gsplat 渲染（480p 直出）→ client 过 HY VAE encoder 得记忆 latent → KV cache / rope 流程**完全不变**
- **Atlas 流**：冻结在最新完成版本 → UI 相机（v1 预设轨道）→ 快照渲染；窗口不可变 + 只追加，天然不阻塞后台更新

`query_pose_mode`:
- `"future"`（v1 默认）：chunk t 自己的 4 个 latent 位姿 + 近历史位姿补足 20 槽（保持 context 长度/rope 布局不变）
- `"fov_kept"`（消融用）：保留原 FOV 选择逻辑选历史位姿，只换内容来源

## 4. 换视角重生成语义（2026-09-21 与用户确认定稿）

**场景**：Atlas 观察中用户转到新视角并从那里继续播放。此时：

- **t-1、t 的内容整体作废**（不只是视频输出）：它们的生成条件包含旧轨迹的上下文，喂进资产就是污染新 rollout 的条件链。资产的可信边界回退到 **t-2 旧轨迹观测**。
- **新视角的参考帧** = 回滚后的资产在新视角位姿下的渲染（`VIEWPOINT_RESET` 原子完成：rollback → 渲染参考帧）。
- **新 rollout 从 0 重新编号**：`TRAJECTORY_RESYNC` 换轨迹表，chunk 从 0 起（配合资产侧 version 守卫，旧 rollout 的在途推送自动被拒）。
- **不换视角 = 什么都不动**：Atlas 观察是只读渲染，观察期间资产更新/DiT 生成照常推进（idea.txt 第 4 点原义）。
- **新视角下的记忆来源** = 复用双工查询路径：新轨迹位姿 → 回滚后资产渲染记忆帧。系统只有一种记忆机制。
- idea.txt 的"不可行"边界精确化为：**换视角"无缝"继续不可行**（硬切可行，本方案即是）；参考帧质量受限于新视角是否被 t-2 观测覆盖。

**实现锚点**：`asset.rollback_to`（deque 回退 + `rejected_stale` 守卫）、`ipc.viewpoint_reset_msg/resync_msg`、`server.handle_viewpoint_reset/resync`、`MemorySource.viewpoint_reset`（generation 计数）、`rollout_fork` B 块 generation 守卫。



| 模块 | 要点 / 风险 |
|---|---|
| `camera_conv.py` | HY 与 MoVieS 内参都是归一化制但公式细节不同——往返一致性单测锁定；SE3 插值（slerp+线性平移） |
| `encoder.py` | 输入 832×480→224 附近（divisor 14）；加载需绕 VGGT-1B 无条件下载 + `./resources` 断言 |
| `asset.py` | 容量=窗口数（默认 20，对齐 memory_frames）；不可变窗口；FIFO v1 |
| `query.py` | 版本过滤（t-2 显式传入）；每个相机按 latent 索引定位窗口；渲染分辨率与编码解耦（480p 直出） |
| `memory_replace.py` | **全项目唯一改 HY 行为的点**：只换记忆 latent 来源，索引选择逻辑第一阶段保留（future 模式换位姿来源，fov_kept 连索引都不换）；monkey-patch 注入，不改仓库文件 |
| `latent_prep.py` | 难点：HY 3D causal VAE latent 形状/数值与 pipeline 内完全对齐（单测：真历史帧 encode vs 原 latent） |
| `server/client.py` | v1 裸 socket + json头 + 二进制体；更新中不锁查询（查旧版本） |

## 5. 里程碑

- **M0** common 层纯函数：相机转换往返、SE3 插值、时间映射单测（含首帧特例边界）
- **M1** encoder + `encode_probe.py`：HY 视频+插值位姿喂 MoVieS，已知视角回投 PSNR（质量底线）
- **M2** asset + query：环形缓冲 + (t,相机) 查询；单测延迟（目标 ≪ 原蒙特卡洛 FOV 检索）与显存
- **M3** `atlas_preview.py`：idea.txt 第 4 点可视化落地
- **M4** 进程化 + IPC：双卡双环境，t-2 版本控制；lookahead 预渲染（DiT 去噪 chunk t 时预渲染 t+1 召回视角，用 t-1 资产版本）
- **M5** hy_client 接入：端到端跑通一个 AR 短例
- **M6** 对比实验：原 FOV 召回 vs 3D 资产召回（质量 + 长视频耗时曲线）；future vs fov_kept 消融

## 6. 第一阶段明确不做

跨窗口高斯融合/去重、换检索索引选择逻辑、换视角**无缝**继续生成（硬切已由第 4 节方案覆盖）、任何训练、对两个仓库的文件级修改。

## 7. 已拍板决策记录

| 日期 | 决策 |
|---|---|
| 2026-09-21 | 高斯方案：MoVieS（替换 StreamSplat/MVSplat） |
| 2026-09-21 | 窗口→全局对齐用解析反变换（canonical 归一化时记录），非估计 |
| 2026-09-21 | 资产 v1 = 窗口环形缓冲（容量 20），不做高斯融合 |
| 2026-09-21 | 查询视角：复用预计算轨迹（future 模式 v1 默认，fov_kept 留作消融） |
| 2026-09-21 | 像素帧位姿 = latent 位姿 SE3 插值（平移线性+旋转 slerp），不用查表复制 |
| 2026-09-21 | 渲染分辨率 480p 直出；进程化 M4 起（此前单进程）；IPC v1 裸 socket |
| 2026-09-21 | 换视角 = 回滚资产至 t-2 + 丢弃 t-1/t + 新视角渲染参考帧 + 从 0 重编号新 rollout（第 4 节） |
