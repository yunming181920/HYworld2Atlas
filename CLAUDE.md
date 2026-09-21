# CLAUDE.md

本目录是一个**研究工作区**（非单一代码库），包含多个第三方仓库 + 一份研究构想（`idea.txt`），目标是把 HY-WorldPlay 的 FOV 历史召回替换为基于流式 4D 高斯的资产式记忆。

> AGENTS.md 是指向本文件的指针；本文件是所有 agent 指南的单一来源，改动只需改这里。
> 方案沿革：StreamSplat（正交相机）→ MVSplat（透视静态）→ **MoVieS（透视 4D 动态，2026-09-21 定稿）**——相机模型与时间维终于都满足需求。

## 项目目标（来自 idea.txt）

1. **问题**：HunyuanWorldplay 的历史召回模块用相机 FOV 检索相近 chunk，视频变长后检索耗时超过生成耗时。像 Helios 那样固定容量的上下文压缩是一种可选策略，本项目选择**用 3D 资产维护历史**，召回时按相机视角从资产中取内容。
2. **手段**：用前馈 4D 高斯方法（MoVieS）维护 3D 资产：单目视频 → 动态高斯（含运动偏移），支持**任意时刻 × 任意透视视角**渲染查询。
3. **架构**：双工管道 —— DiT（去噪）在一张卡上，decoder / 高斯 / encoder 在另一张卡上，替代原来的上下文。此时 t 时刻的 DiT 刚完成 t-1 阶段的生成，能参考的只有 **t-2 阶段更新完成的 3D 资产**。
4. **附加目标（模拟 Atlas）**：时间静止时从任意相机视角观察 3D 资产。最简单的实现：t-1 阶段生成完成后，3D 资产仍是 t-2 更新后的版本，只展示 t-2 阶段内容；用户观察期间，后台继续 t-1 阶段的资产更新和 t 阶段的 DiT 生成。注意：**换视角后继续生成不经训练做不到**。

## 目录结构

| 路径 | 说明 |
|---|---|
| `idea.txt` | 研究构想原文（中文），本项目的出发点 |
| `HY-WorldPlay/` | 腾讯混元 HY-World 1.5 交互式世界模型（基于 HunyuanVideo-1.5 DiT） |
| `MoVieS/` | 前馈 4D 动态高斯：单目视频 → 外观/几何/运动联合重建（CVPR 2026, arXiv:2507.10065）**当前方案** |
| `StreamSplat/` | （已放弃，参考）流式动态 3DGS，正交相机 |

## HY-WorldPlay 关键代码地图

- **推理入口**：`run.sh` → `hyvideo/generate.py`（`main()` :829，`generate_video()` :700）。`pose_to_input()` :173 把相机轨迹转成 `viewmats`/`Ks`/`action` 三元组。
- **AR 主循环**：`hyvideo/pipelines/worldplay_video_pipeline.py` 的 `ar_rollout` :1001 → `_ar_rollout_inner` :1042，逐 chunk 去噪，**1 chunk = 4 latents = 16 帧**。
- **历史召回（本项目要替换的部分）**：`hyvideo/utils/retrieval_context.py`
  - `select_aligned_memory_frames()` :216 —— 核心入口，选记忆帧索引
  - `calculate_fov_overlap_similarity()` :139 —— 蒙特卡洛采样的 FOV 重叠相似度
  - 调用位置：`worldplay_video_pipeline.py:1125`（AR）、:1330（bi）；参数 `memory_frames=20, temporal_context_size=12, pred_latent_size=4`
  - 训练侧在 `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py:612` 有同一套逻辑的拷贝，**训练/推理共享 FOV 检索逻辑，改动时两边要同步**
- **DiT**：`hyvideo/models/transformers/worldplay_1_5_transformer.py`（`HunyuanVideo_1_5_DiffusionTransformer` :739）；相机位姿经 ProPE 注入 QKV：`hyvideo/prope/camera_rope.py`
- **VAE**：`hyvideo/models/autoencoders/hunyuanvideo_15_vae_w_cache.py`（带 feature cache，支持流式 chunk 解码）
- **KV cache**：视觉上下文 KV 每 chunk 重算（`init_kv_cache` pipeline:989），文本 KV 只算一次
- **训练**：`scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh` → `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`
- **模型下载**：`download_models.py`（HF gated 的 SigLIP 需要 token，`--skip_vision_encoder` 可跳过）
- **相机约定**：`viewmats` 为 W2C `(1, num_latents, 4, 4)`；`Ks` 归一化内参（fx,fy 除以 2×cx,cy，主点归一到 0.5，generate.py:217-221）——与 MoVieS 的归一化 fxfycxcy 约定同类（主点约 0.5），对接时统一即可

## MoVieS 关键代码地图

- **定位**：前馈 4D 重建（1 秒级）。单目视频 + 相机位姿 + 时间戳 → 逐像素高斯（外观/几何/深度置信度）+ **时间条件化运动偏移**，支持任意 (t, camera) 渲染。
- **模型构建**（`src/infer_davis_nvs.py:38-42` 是手工推理的最小范例）：
  - `opt = opt_dict["movies"]`（`src/options.py:234-242`）→ `SplatRecon(opt, load_lpips=False)`（`src/models/splatrecon.py:20`，`load_lpips=False` 避开 LPIPS）
  - 推理只用两个成员：`model.backbone` = `VGGSplaT`（`src/models/networks/vggsplat.py:21`）+ `model.gs_renderer` = `GaussianRenderer`（`src/models/gs_render/gs_renderer.py:14`）；`SplatRecon.forward` 是训练 loss 接口，推理不走
  - ckpt：HF `chenguolin/MoVieS` 的 `movies_ckpt.safetensors`，`load_state_dict(..., strict=True)` 全量覆盖
- **backbone 前向签名**（`vggsplat.py:153-161`）：
  `forward(images (B,F_in,3,H,W), C2W (B,F_in,4,4), fxfycxcy (B,F_in,4), input_timesteps (B,F_in), output_timesteps (B,F_out), frames_chunk_size=8)`
  → `model_outputs`（逐像素属性 dict，`(B,F_in,C,H,W)`：depth/depth_conf/color/scale/rotation/opacity/conf）+ `pred_motions (B,F_out,F_in,3+1,H,W)`（每个输出时刻、每个输入帧像素的运动偏移+置信度）+ `pred_motion_splat`（时间条件属性残差，运动大区域生效）
- **渲染签名**（`gs_renderer.py:25-40`）：`render(model_outputs, input_C2W, input_fxfycxcy, C2W, fxfycxcy, height, width, ...) → {image, alpha, depth, ...}`。**时间不是渲染器输入**：任意 (t, camera) 渲染 = 把 `pred_motions[:, i]` 写入 `model_outputs["offset"]` + `update(pred_motion_splat[i])` 再 render（`infer_davis_nvs.py:84-96` 范例）。`output_timesteps` 支持任意连续值（sin-cos 编码），新时刻需重新过 DPT 头。
- **相机格式**：extrinsics 为 **C2W** (B,F,4,4)，OpenCV 约定；内参为归一化 `fxfycxcy`（[0,1]）。训练数据管线把首相机归一到单位阵 + 尺度归一（`src/data/base_dataset.py:215-231` canonical 约定）；DAVIS 的 npz 位姿由 MegaSAM 估计、已按 VGGT 风格归一。几何工具集中在 `src/utils/geo_util.py`（`inverse_c2w`/`unproject_depth`/`plucker_ray`/`fxfycxcy_to_intrinsics`）。
- **渲染器**：**gsplat v1.5.0**（nerfstudio-project，pip 源码编译 CUDA）——透视投影，`viewmats` W2C + 像素 K（归一化 fxfycxcy × 渲染宽高，`gs_render/gs_util.py:42-47`）；渲染分辨率经 `render(height, width)` 与输入解耦。`extensions/` 只有运维脚本，**无需编译**；`extensions/vggt` 需按 `settings/setup.sh:20` 自行 clone。
- **流式能力**：帧数可变（VGGT 交替注意力 + random_video_size，min 2 帧；DAVIS 推理一次喂 13 帧），**但无跨窗口状态/对齐机制**——各窗口输出高斯在各自的 canonical 坐标系，窗口间位姿/尺度对齐需调用方自建。
- **动态建模**：无 velocity/轨迹/变形场——运动 = per-timestep 偏移 + 属性残差（`vggsplat.py:304-314` 运动头、:123-143 motion_splat 头，mask_by_motion=0.005）。znear/zfar = 0.001/100。输入 224×224 训练（size_divisor=14，DINOv2 patch），推理分辨率由数据决定。

## ⚠️ 关键技术约束（跨库对接时必读）

1. **坐标系/尺度对齐是头号工程问题**：MoVieS 每个窗口独立做"首相机归一 + 尺度归一"（canonical 系）；HY 的轨迹有绝对尺度（w=0.08/帧）。双工管道需要自建**窗口间对齐**（把各窗口 canonical 高斯统一到 HY 全局系：利用窗口重叠帧的位姿 + 尺度因子）。
2. **时间戳约定**：MoVieS 时间按窗口归一 [0,1]；HY 的时间是全局 chunk 序号。资产的时间轴需要全局化映射。
3. **帧数窗口**：MoVieS 可变帧数（DAVIS 13 帧），但运动偏移是**窗口内时间条件化**的——长视频的"任意时刻查询"需要把全局时刻映射回某个窗口的输出时刻；跨窗口的运动连续性无保证。idea.txt 的"流式维护 3D 资产"仍需外层增量机制（逐 chunk 编码 → 并入全局资产 → 按需查询），MoVieS 提供了动态建模但**没提供资产融合**。
4. **分辨率匹配**：MoVieS 训练 224×224（divisor 14）；HY 输出 832×480。编码时降采样到 224 附近（召回资产允许模糊），渲染查询分辨率任意（光栅化与输入解耦）。
5. **双工时序约束**：t 时刻 DiT 只能用 t-2 阶段资产（DiT 刚完成 t-1 生成，资产更新晚一拍）。设计召回接口时按这个滞后建模。
6. **训练/推理一致性**：改 HY 的召回逻辑时，`hyvideo/utils/retrieval_context.py` 与 `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py` 里的拷贝要同步。
7. **位姿估计**：MoVieS DAVIS 流程的位姿来自 MegaSAM；HY 自带位姿（`viewmats`），**不需要额外估计**，但要确认位姿质量（长视频漂移会直接进资产）。

## 当前阶段与技术路线

项目处于**方案选定（MoVieS）、尚未写实验代码**的阶段。预期的双工管道架构：

```
┌─────────────── GPU A ───────────────┐      ┌─────────────── GPU B ───────────────┐
│  HY-WorldPlay DiT                    │      │  VAE decoder → MoVieS backbone      │
│  (t 阶段去噪)                        │ ───► │  → 全局 4D 高斯资产 (t-1 阶段更新)  │
│  召回时查询: t-2 阶段资产             │ ◄─── │  视角查询: (t, 相机) 透视渲染       │
└──────────────────────────────────────┘      └─────────────────────────────────────┘
```

三个待解决的核心问题（按优先级）：

1. **窗口对齐与全局资产维护**：逐 chunk/窗口用 MoVieS 编码（各在 canonical 系）→ 位姿+尺度对齐到 HY 全局系 → 并入全局缓冲（新数据替换旧高斯、容量上限对标 Helios）。MoVieS 没有现成的融合机制，这是自研重点。
2. **视角查询接口**：封装"给定 (t, c2w, 归一化 K)，从全局资产渲染 RGB/深度"：把全局 t 映射到窗口时间轴 → 取该窗口 model_outputs + 运动偏移 → gsplat 渲染。渲染分辨率独立于编码分辨率。
3. **召回内容进 DiT**：渲染出的 RGB/深度需变回 DiT 能吃的上下文（经 HY 的 VAE encoder 回 latent，兼容 KV cache 流程），替换 `select_aligned_memory_frames` 的索引选择路径。

## 已知坑

- **SigLIP 权重是 HF gated** 的，`download_models.py` 需要 token；跑不通时可用 `--skip_vision_encoder` 先绕过。
- **MoVieS import 链拉训练栈**：`import src.models` → diffusers；`src/utils` → wandb/accelerate/matplotlib（`vis_util.py`/`util.py`/`track_visualizer.py`）。推理需要装这些包，但不需要 accelerate launch/训练数据。精简部署可 fork 掉这些 import。
- **MoVieS 构建时无条件 `VGGT.from_pretrained("facebook/VGGT-1B")` 下载**（`vggsplat.py:30`），随后被 ckpt 全量覆盖——纯推理也需 HF 网络/缓存（绕开需改代码）。
- **MoVieS 的 `src/options.py` 导入时 assert `./resources` 目录存在**（:153-155, :232-242）——工作目录必须在仓库根。
- **gsplat v1.5.0 + torch-scatter 需源码编译**，与 MoVieS 一起用 torch 2.5.1+cu124（`settings/setup.sh`）。
- **三个环境建议彼此独立**：HY / MoVieS / StreamSplat（仅参考）各自 conda 环境，实验代码通过进程间通信（文件/共享内存/socket）桥接双工管道。
- **换视角继续生成不可行**（idea.txt 明确指出，未经训练做不到）——Atlas 式自由观察只在"时间静止"时提供，不要试图实现换视角续生成，除非后续明确加入训练计划。

## 验证路径

实现过程中每一步的可验证中间产物：

1. 跑通 MoVieS 官方 DAVIS 推理（`infer_davis_nvs.py`，需 HF 下载 ckpt + DAVIS npz），确认 gsplat 编译成功。
2. 手工构建剥离版（`opt_dict["movies"]` + `SplatRecon(load_lpips=False)` + safetensors 加载），验证 backbone + renderer 独立于训练框架可用。
3. 用 HY 生成的视频帧 + HY 位姿（w2c→c2w、Ks→fxfycxcy）喂 MoVieS：验证重建质量（已知视角回投误差 + 邻近视角渲染对比原视频）。
4. 实现窗口对齐 + 全局资产缓冲 + (t, 相机) 查询接口，单测：给定轨迹位姿，查询延迟应远低于原 FOV 检索。
5. 接入双工管道，端到端对比：原版 FOV 召回 vs 4D 资产召回的生成质量与长视频耗时曲线。

## 常用命令

```bash
# HY-WorldPlay 推理（先填 run.sh 里的 MODEL_PATH 等环境变量，或走 download_models.py）
cd HY-WorldPlay && bash run.sh          # 默认启用 AR 蒸馏 4 步模型；bi / ar 版本在注释里切换

# MoVieS DAVIS 新视角合成推理（需先 HF 下载 ckpt + DAVIS 数据到 resources/）
cd MoVieS && python src/infer_davis_nvs.py --name motocross-bumps
```

## 约定（agent 工作守则）

- 本目录**不是 git 仓库**；子目录是独立克隆的第三方项目，不要往它们里面提交改动，自己的实验代码放工作区根目录新建的模块里。小范围必要的实验性修改允许，但要保持可回滚、可辨认（注释标明用途）。
- 与用户沟通、写文档用中文；代码注释跟随所在仓库的风格。
- 动手前先读 `idea.txt` 和本文件——所有设计决策以 idea.txt 的四点构想为准，特别是"t 时刻只能用 t-2 资产"的滞后约束。
- 改 HY 的召回逻辑时，检查推理侧与训练侧两份拷贝是否需要同步（见"关键技术约束"第 6 条）。
