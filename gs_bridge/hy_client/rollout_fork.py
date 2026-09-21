"""Forked HY AR rollout with the gs_bridge memory source swapped in (M5).

Direct copy of `HunyuanVideo_1_5_Pipeline._ar_rollout_inner`
(pipelines/worldplay_video_pipeline.py:1042-1303) with exactly TWO changed blocks
(marked `# === gs_bridge`):

    A. after HY's FOV selection computes `selected_frame_indices`: optionally
       override the index list (query_pose_mode="future") and swap
       `context_latents` from rollout history -> asset-rendered latents
       (version-capped at t-1, i.e. idea.txt's "t-2 asset").
       Falls back to the original history latents whenever the asset cannot
       serve (early chunks / empty cap / render failure) -- the rollout NEVER
       breaks because of the bridge.

    B. after each chunk finishes denoising: push the chunk's decoded frames to
       the asset server (CHUNK_UPDATE). Decode happens here so the update
       overlaps the next chunk's selection (the server encodes while the DiT
       denoises chunk t+1 -- the duplex pipeline).

Everything else (kv-cache, rope offsets, CFG, scheduler) is byte-identical to the
original, so DiT behavior changes ONLY through the context content.
"""

import numpy as np
import torch

from hyvideo.pipelines.worldplay_video_pipeline import (
    auto_offload_model,
    select_aligned_memory_frames,
)


def ar_rollout_inner_forked(
    self,
    latents,
    timesteps,
    prompt_embeds,
    prompt_mask,
    vision_states,
    cond_latents,
    task_type,
    extra_kwargs,
    viewmats,
    Ks,
    action,
    device,
):
    bridge = getattr(self, "gs_bridge", None)  # MemorySource | None
    use_bridge = bridge is not None and bridge.enabled
    rollout_generation = bridge.generation if use_bridge else 0  # reset ordering guard

    self.init_kv_cache()
    positive_idx = 1 if self.do_classifier_free_guidance else 0
    stabilization_level = 15
    # text, siglip, byt5 embedding cache
    with (
        torch.autocast(
            device_type="cuda",
            dtype=self.target_dtype,
            enabled=self.autocast_enabled,
        ),
        auto_offload_model(
            self.transformer, self.execution_device, enabled=self.enable_offloading
        ),
    ):
        extra_kwargs_pos = {
            "byt5_text_states": extra_kwargs["byt5_text_states"][
                positive_idx, None, ...
            ],
            "byt5_text_mask": extra_kwargs["byt5_text_mask"][
                positive_idx, None, ...
            ],
        }
        t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
        self._kv_cache = self.transformer(
            bi_inference=False,
            ar_txt_inference=True,
            ar_vision_inference=False,
            timestep_txt=t_expand_txt,
            text_states=prompt_embeds[positive_idx, None, ...],
            encoder_attention_mask=prompt_mask[positive_idx, None, ...],
            vision_states=vision_states[positive_idx, None, ...],
            mask_type=task_type,
            extra_kwargs=extra_kwargs_pos,
            kv_cache=self._kv_cache,
            cache_txt=True,
        )
        if self.do_classifier_free_guidance:
            extra_kwargs_neg = {
                "byt5_text_states": extra_kwargs["byt5_text_states"][0, None, ...],
                "byt5_text_mask": extra_kwargs["byt5_text_mask"][0, None, ...],
            }
            t_expand_txt = torch.tensor([0]).to(device).to(latents.dtype)
            self._kv_cache_neg = self.transformer(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=t_expand_txt,
                text_states=prompt_embeds[0, None, ...],
                encoder_attention_mask=prompt_mask[0, None, ...],
                vision_states=vision_states[0, None, ...],
                mask_type=task_type,
                extra_kwargs=extra_kwargs_neg,
                kv_cache=self._kv_cache_neg,
                cache_txt=True,
            )

    selected_frame_indices = []

    for chunk_i in range(self.chunk_num):
        if chunk_i > 0:
            current_frame_idx = (
                chunk_i * self.chunk_latent_frames
            )  # the current frame index to generate

            selected_frame_indices = []
            for chunk_start_idx in range(
                current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4
            ):
                selected_history_frame_id = select_aligned_memory_frames(
                    viewmats[0].cpu().detach().numpy(),
                    chunk_start_idx,
                    memory_frames=20,
                    temporal_context_size=12,
                    pred_latent_size=4,
                    points_local=self.points_local,
                    device=device,
                )
                selected_frame_indices += selected_history_frame_id
            selected_frame_indices = sorted(list(set(selected_frame_indices)))
            to_remove = list(
                range(
                    current_frame_idx, current_frame_idx + self.chunk_latent_frames
                )
            )
            selected_frame_indices = [
                x for x in selected_frame_indices if x not in to_remove
            ]

            # === gs_bridge [A]: memory source swap ===========================
            bridge_latents = None
            if use_bridge:
                try:
                    query_indices = bridge.query_pose_latents(
                        chunk_i, selected_frame_indices
                    )
                    bridge_latents = bridge.render_memory_latents(
                        chunk_i, query_indices, device, latents.dtype
                    )
                    if bridge_latents is not None:
                        if bridge.mode == "future":
                            # index swap: rope/kv layout follows the future indices
                            selected_frame_indices = list(query_indices)
                        # shape guard: rendered latents must match index count
                        if bridge_latents.shape[2] != len(selected_frame_indices):
                            bridge_latents = bridge_latents[:, :, : len(selected_frame_indices)]
                except Exception as e:  # never break the rollout
                    print(f"[gs_bridge] render failed, fallback to history: {e}")
                    bridge.fallback_count += 1
                    bridge_latents = None
            # ================================================================

            if bridge_latents is not None:
                context_latents = bridge_latents.to(latents.dtype)
            else:
                context_latents = latents[:, :, selected_frame_indices]
            context_cond_latents_input = cond_latents[:, :, selected_frame_indices]
            context_latents_input = torch.concat(
                [context_latents, context_cond_latents_input], dim=1
            )

            context_viewmats = viewmats[:, selected_frame_indices].to(device)
            context_Ks = Ks[:, selected_frame_indices].to(device)
            context_action = action[:, selected_frame_indices].to(device)

            context_timestep = torch.full(
                (len(selected_frame_indices),),
                stabilization_level - 1,
                device=device,
                dtype=timesteps.dtype,
            )
            # compute kv cache
            with (
                torch.autocast(
                    device_type="cuda",
                    dtype=self.target_dtype,
                    enabled=self.autocast_enabled,
                ),
                auto_offload_model(
                    self.transformer,
                    self.execution_device,
                    enabled=self.enable_offloading,
                ),
            ):
                self._kv_cache = self.transformer(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=True,
                    hidden_states=context_latents_input,
                    timestep=context_timestep,
                    timestep_r=None,
                    mask_type=task_type,
                    return_dict=False,
                    viewmats=context_viewmats.to(self.target_dtype),
                    Ks=context_Ks.to(self.target_dtype),
                    action=context_action.to(self.target_dtype),
                    kv_cache=self._kv_cache,
                    cache_vision=True,
                    rope_temporal_size=context_latents_input.shape[2],
                    start_rope_start_idx=0,
                )
                if self.do_classifier_free_guidance:
                    self._kv_cache_neg = self.transformer(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=context_latents_input,
                        timestep=context_timestep,
                        timestep_r=None,
                        mask_type=task_type,
                        return_dict=False,
                        viewmats=context_viewmats.to(self.target_dtype),
                        Ks=context_Ks.to(self.target_dtype),
                        action=context_action.to(self.target_dtype),
                        kv_cache=self._kv_cache_neg,
                        cache_vision=True,
                        rope_temporal_size=context_latents_input.shape[2],
                        start_rope_start_idx=0,
                    )

            self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        start_idx = chunk_i * self.chunk_latent_frames
        end_idx = chunk_i * self.chunk_latent_frames + self.chunk_latent_frames

        with (
            self.progress_bar(total=self.num_inference_steps) as progress_bar,
            auto_offload_model(
                self.transformer,
                self.execution_device,
                enabled=self.enable_offloading,
            ),
        ):
            for i, t in enumerate(timesteps):
                timestep_input = torch.full(
                    (self.chunk_latent_frames,),
                    t,
                    device=device,
                    dtype=timesteps.dtype,
                )
                latent_model_input = latents[:, :, start_idx:end_idx]
                cond_latents_input = cond_latents[:, :, start_idx:end_idx]

                viewmats_input = viewmats[:, start_idx:end_idx].to(device)
                Ks_input = Ks[:, start_idx:end_idx].to(device)
                action_input = action[:, start_idx:end_idx].to(device)

                latents_concat = torch.concat(
                    [latent_model_input, cond_latents_input], dim=1
                )
                latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                with torch.autocast(
                    device_type="cuda",
                    dtype=self.target_dtype,
                    enabled=self.autocast_enabled,
                ):
                    noise_pred = self.transformer(
                        bi_inference=False,
                        ar_txt_inference=False,
                        ar_vision_inference=True,
                        hidden_states=latents_concat,
                        timestep=timestep_input,
                        timestep_r=None,
                        mask_type=task_type,
                        return_dict=False,
                        viewmats=viewmats_input.to(self.target_dtype),
                        Ks=Ks_input.to(self.target_dtype),
                        action=action_input.to(self.target_dtype),
                        kv_cache=self._kv_cache,
                        cache_vision=False,
                        rope_temporal_size=latents_concat.shape[2]
                        + len(selected_frame_indices),
                        start_rope_start_idx=len(selected_frame_indices),
                    )[0]
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond = self.transformer(
                            bi_inference=False,
                            ar_txt_inference=False,
                            ar_vision_inference=True,
                            hidden_states=latents_concat,
                            timestep=timestep_input,
                            timestep_r=None,
                            mask_type=task_type,
                            return_dict=False,
                            viewmats=viewmats_input.to(self.target_dtype),
                            Ks=Ks_input.to(self.target_dtype),
                            action=action_input.to(self.target_dtype),
                            kv_cache=self._kv_cache_neg,
                            cache_vision=False,
                            rope_temporal_size=latents_concat.shape[2]
                            + len(selected_frame_indices),
                            start_rope_start_idx=len(selected_frame_indices),
                        )[0]

                if self.do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )

                latent_model_input = self.scheduler.step(
                    noise_pred, t, latent_model_input, return_dict=False
                )[0]
                latents[:, :, start_idx:end_idx] = latent_model_input[
                    :, :, -self.chunk_latent_frames :
                ]

                if i == len(timesteps) - 1 or (
                    (i + 1) > self.num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()

        # === gs_bridge [B]: duplex update -- push this chunk to the asset ===
        # Guard 1 (in-flight reset): if a viewpoint reset happened since this rollout
        # started (bridge.generation > rollout's generation), the rollout is abandoned
        # -- stop pushing; its remaining output is old-trajectory content.
        # Guard 2 (stale insert): even if a push already left the socket, the server's
        # asset rejects version <= current after rollback, so reordering is safe.
        if use_bridge and bridge.generation == rollout_generation:
            try:
                bridge.push_finished_chunk(self, chunk_i, latents, device)
            except Exception as e:
                print(f"[gs_bridge] chunk push failed (chunk {chunk_i}): {e}")
        # ====================================================================

    return latents
