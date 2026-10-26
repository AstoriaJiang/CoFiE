import json
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    create_causal_mask = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
except ImportError:
    Qwen3VLCausalLMOutputWithPast = None


class Qwen3VLForConditionalGenerationDTD(Qwen3VLForConditionalGeneration):

    def _ensure_stage_timers(self):
        if not hasattr(self, "_stage_timing"):
            self._stage_timing = {
                "visual_encode": 0.0,
                "token_drop": 0.0,
                "llm_prefill": 0.0,
                "llm_decode": 0.0,
            }

    def reset_stage_timing(self):
        self._stage_timing = {
            "visual_encode": 0.0,
            "token_drop": 0.0,
            "llm_prefill": 0.0,
            "llm_decode": 0.0,
        }

    def get_stage_timing(self):
        self._ensure_stage_timers()
        return dict(self._stage_timing)

    @staticmethod
    def _is_prefill_step(past_key_values):
        if past_key_values is None:
            return True
        if hasattr(past_key_values, 'get_seq_length'):
            return past_key_values.get_seq_length() == 0
        return True

    def token_drop(
        self,
        method: str,
        threshold: float,
        absolute: bool = True,
        hidden_states: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
        dr_save_path: Optional[str] = None,
        dp_save_path: Optional[str] = None,
        sim_save_path: Optional[str] = None,
    ):
        if absolute is None:
            absolute = True
        if method not in ['pixel', 'feature']:
            raise ValueError(f"Method {method} is not supported for token dropping.")

        batch_size, seq_len, _ = hidden_states.size()

        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id

        if method == 'pixel':
            grid_t, grid_h, grid_w = video_grid_thw[0]
            merge_size = self.config.vision_config.spatial_merge_size
            channel = self.config.vision_config.in_channels
            temporal_patch_size = self.config.vision_config.temporal_patch_size
            patch_size = self.config.vision_config.patch_size
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]
            pv = pixel_values_videos.view(
                grid_t, grid_h // merge_size, grid_w // merge_size,
                merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size
            )
            pv = pv.permute(0, 6, 1, 2, 5, 3, 4, 7, 8)
            pv = pv.reshape(
                grid_t, temporal_patch_size,
                (grid_h // merge_size) * (grid_w // merge_size),
                channel, -1
            )
            for c in range(channel):
                pv[:, :, :, c] = pv[:, :, :, c] * OPENAI_CLIP_STD[c] + OPENAI_CLIP_MEAN[c]

        cos, sin = position_embeddings
        cos = cos.to(hidden_states.device)
        sin = sin.to(hidden_states.device)
        dropped_hidden_list, dropped_cos_list, dropped_sin_list = [], [], []
        dropped_pos_ids_list, dropped_vpm_list = [], []
        visual_keep_masks = []
        cnt_total, cnt_drop = 0, 0
        dropped_positions_info = []
        similarity_info_list = []

        for i in range(batch_size):
            sample_input_ids = input_ids[i]
            vis_start = (sample_input_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
            vis_end = (sample_input_ids == vision_end_token_id).nonzero(as_tuple=True)[0]
            visual_token_indices = set()
            for si, ei in zip(vis_start, vis_end):
                visual_token_indices.update(range(si.item() + 1, ei.item()))

            sample_pos_ids = position_ids[:, i].to(hidden_states.device)
            time_ids = sample_pos_ids[0]
            height_ids = sample_pos_ids[1]
            width_ids = sample_pos_ids[2]
            unique_times = time_ids.unique()

            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
            drop_pos = {}
            sample_sim_info = {"method": method, "frames": []}

            if len(visual_token_indices) > 0:
                if method == 'pixel':
                    prev_frame_pv = None
                    idx_this_frame = -1
                    for t in unique_times:
                        frame_indices = (time_ids == t).nonzero(as_tuple=True)[0]
                        frame_indices = [idx.item() for idx in frame_indices if idx.item() in visual_token_indices]
                        if len(frame_indices) == 0:
                            continue
                        frame_indices = torch.tensor(frame_indices, dtype=torch.long, device=hidden_states.device)
                        idx_this_frame += 1
                        start_idx = frame_indices[0]
                        h_start, w_start = height_ids[start_idx].item(), width_ids[start_idx].item()
                        curr_frame_pv = pv[idx_this_frame, -1]
                        cnt_total += len(frame_indices)
                        if prev_frame_pv is None:
                            prev_frame_pv = pv[idx_this_frame, 0]
                            continue
                        difference = torch.abs(prev_frame_pv - curr_frame_pv).mean(dim=2).mean(dim=1)
                        if absolute:
                            drop_idx = (difference < threshold).nonzero(as_tuple=True)[0].to(start_idx.device) + start_idx
                        else:
                            k = int(difference.numel() * threshold)
                            drop_idx = torch.topk(difference, k, largest=False).indices.to(start_idx.device) + start_idx
                        cnt_drop += len(drop_idx)
                        prev_frame_pv = pv[idx_this_frame, 0]
                        keep_mask[drop_idx] = False
                        for di in drop_idx:
                            di_ = di.item()
                            h_ = height_ids[di_].item() - h_start
                            w_ = width_ids[di_].item() - w_start
                            drop_pos.setdefault(idx_this_frame, []).append((h_, w_))

                elif method == 'feature':
                    prev_frame_feat = None
                    idx_this_frame = -1
                    for t in unique_times:
                        frame_indices = (time_ids == t).nonzero(as_tuple=True)[0]
                        frame_indices = [idx.item() for idx in frame_indices if idx.item() in visual_token_indices]
                        if len(frame_indices) == 0:
                            continue
                        frame_indices = torch.tensor(frame_indices, dtype=torch.long, device=hidden_states.device)
                        idx_this_frame += 1
                        start_idx = frame_indices[0]
                        h_start, w_start = height_ids[start_idx].item(), width_ids[start_idx].item()
                        curr_frame_feat = hidden_states[i, frame_indices]
                        rel_h = [height_ids[idx.item()].item() - h_start for idx in frame_indices]
                        rel_w = [width_ids[idx.item()].item() - w_start for idx in frame_indices]
                        grid_h_cur = max(rel_h) + 1
                        grid_w_cur = max(rel_w) + 1
                        cnt_total += len(frame_indices)
                        if prev_frame_feat is None:
                            prev_frame_feat = curr_frame_feat
                            sample_sim_info["frames"].append({
                                "frame_index": idx_this_frame,
                                "grid_h": grid_h_cur, "grid_w": grid_w_cur,
                                "similarity_map": None,
                            })
                            continue
                        similarity = F.cosine_similarity(prev_frame_feat, curr_frame_feat, dim=1)
                        sim_map = [[None] * grid_w_cur for _ in range(grid_h_cur)]
                        for h_, w_, s_ in zip(rel_h, rel_w, similarity.detach().float().cpu().tolist()):
                            sim_map[h_][w_] = s_
                        sample_sim_info["frames"].append({
                            "frame_index": idx_this_frame,
                            "grid_h": grid_h_cur, "grid_w": grid_w_cur,
                            "similarity_map": sim_map,
                        })
                        if absolute:
                            drop_idx = (similarity > threshold).nonzero(as_tuple=True)[0].to(start_idx.device) + start_idx
                        else:
                            k = int(similarity.numel() * threshold)
                            drop_idx = torch.topk(similarity, k).indices.to(start_idx.device) + start_idx
                        cnt_drop += len(drop_idx)
                        prev_frame_feat = curr_frame_feat
                        keep_mask[drop_idx] = False
                        for di in drop_idx:
                            di_ = di.item()
                            h_ = height_ids[di_].item() - h_start
                            w_ = width_ids[di_].item() - w_start
                            drop_pos.setdefault(idx_this_frame, []).append((h_, w_))

            dropped_positions_info.append(drop_pos)
            similarity_info_list.append(sample_sim_info)

            dev = hidden_states.device
            dropped_hidden_list.append(hidden_states[i][keep_mask.to(dev)].unsqueeze(0))
            dropped_cos_list.append(cos[i][keep_mask.to(dev)].unsqueeze(0))
            dropped_sin_list.append(sin[i][keep_mask.to(dev)].unsqueeze(0))
            dropped_pos_ids_list.append(sample_pos_ids[:, keep_mask.to(dev)].unsqueeze(1))

            if visual_pos_masks is not None:
                old_vpm = visual_pos_masks[i].to(dev)
                visual_keep = keep_mask.to(dev)[old_vpm]
                visual_keep_masks.append(visual_keep)
                dropped_vpm_list.append(old_vpm[keep_mask.to(dev)].unsqueeze(0))

        dropped_hidden = torch.cat(dropped_hidden_list, dim=0)
        dropped_cos = torch.cat(dropped_cos_list, dim=0)
        dropped_sin = torch.cat(dropped_sin_list, dim=0)
        dropped_pos_ids = torch.cat(dropped_pos_ids_list, dim=1)
        dropped_pos_emb = (dropped_cos, dropped_sin)

        dropped_vpm = None
        dropped_deepstack = None
        if visual_pos_masks is not None and len(dropped_vpm_list) > 0:
            dropped_vpm = torch.cat(dropped_vpm_list, dim=0)
            if deepstack_visual_embeds is not None and len(visual_keep_masks) > 0:
                vk = torch.cat(visual_keep_masks, dim=0) if batch_size > 1 else visual_keep_masks[0]
                dropped_deepstack = [emb[vk.to(emb.device)] for emb in deepstack_visual_embeds]

        if dr_save_path and cnt_total > 0:
            info = {'drop': cnt_drop, 'total': cnt_total, 'ratio': cnt_drop / cnt_total}
            with open(dr_save_path, 'a' if os.path.exists(dr_save_path) else 'w') as f:
                f.write(json.dumps(info) + '\n')
        if dp_save_path:
            with open(dp_save_path, 'a' if os.path.exists(dp_save_path) else 'w') as f:
                f.write(json.dumps(dropped_positions_info) + '\n')
        if sim_save_path:
            with open(sim_save_path, 'a' if os.path.exists(sim_save_path) else 'w') as f:
                f.write(json.dumps(similarity_info_list) + '\n')

        return dropped_hidden, dropped_pos_emb, dropped_pos_ids, dropped_vpm, dropped_deepstack

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: int = 0,
        drop_method: Optional[str] = None,
        drop_threshold: Optional[float] = None,
        drop_absolute: Optional[bool] = None,
        dr_save_path: Optional[str] = None,
        dp_save_path: Optional[str] = None,
        sim_save_path: Optional[str] = None,
        **kwargs,
    ):
        self._ensure_stage_timers()
        model = self.model

        should_drop = (
            drop_method is not None
            and drop_method.lower() != 'none'
            and pixel_values_videos is not None
        )

        if not should_drop:
            lm_key = "llm_prefill" if self._is_prefill_step(past_key_values) else "llm_decode"

            if pixel_values_videos is not None:
                orig_get_video_features = model.get_video_features
                visual_before = self._stage_timing["visual_encode"]

                def _timed_get_video_features(*args, **inner_kwargs):
                    visual_t0 = time.perf_counter()
                    out = orig_get_video_features(*args, **inner_kwargs)
                    self._stage_timing["visual_encode"] += time.perf_counter() - visual_t0
                    return out

                model.get_video_features = _timed_get_video_features
                try:
                    fwd_t0 = time.perf_counter()
                    result = super().forward(
                        input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels,
                        pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                        mm_token_type_ids=mm_token_type_ids, logits_to_keep=logits_to_keep,
                        **kwargs,
                    )
                    fwd_elapsed = time.perf_counter() - fwd_t0
                finally:
                    model.get_video_features = orig_get_video_features

                visual_elapsed = self._stage_timing["visual_encode"] - visual_before
                self._stage_timing[lm_key] += max(0.0, fwd_elapsed - visual_elapsed)
                return result

            fwd_t0 = time.perf_counter()
            result = super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids, logits_to_keep=logits_to_keep,
                **kwargs,
            )
            self._stage_timing[lm_key] += time.perf_counter() - fwd_t0
            return result

        # ===== Token drop path: replicate Qwen3VLModel + Qwen3VLTextModel forward =====
        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            visual_t0 = time.perf_counter()
            video_outputs = model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            self._stage_timing["visual_encode"] += time.perf_counter() - visual_t0
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_emb, vid_emb in zip(deepstack_image_embeds, deepstack_video_embeds):
                joint = img_emb.new_zeros(visual_pos_masks.sum(), img_emb.shape[-1]).to(img_emb.device)
                joint[image_mask_joint, :] = img_emb
                joint[video_mask_joint, :] = vid_emb
                deepstack_visual_embeds.append(joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        # --- Qwen3VLTextModel forward with token_drop ---
        language_model = model.language_model
        use_cache = kwargs.pop('use_cache', None)

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=language_model.config)

        if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            vision_position_ids = position_ids[1:]
        elif position_ids is not None and position_ids.ndim == 2:
            pos4 = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
            text_position_ids = pos4[0]
            vision_position_ids = pos4[1:]
        else:
            text_position_ids = None
            vision_position_ids = position_ids

        hidden_states = inputs_embeds
        position_embeddings = language_model.rotary_emb(hidden_states, vision_position_ids)

        if self.config.vision_start_token_id in input_ids[0]:
            token_drop_t0 = time.perf_counter()
            hidden_states, position_embeddings, vision_position_ids, visual_pos_masks, deepstack_visual_embeds = \
                self.token_drop(
                    method=drop_method, threshold=drop_threshold, absolute=drop_absolute,
                    hidden_states=hidden_states, pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw, position_embeddings=position_embeddings,
                    position_ids=vision_position_ids, input_ids=input_ids,
                    visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds,
                    dr_save_path=dr_save_path, dp_save_path=dp_save_path, sim_save_path=sim_save_path,
                )
            self._stage_timing["token_drop"] += time.perf_counter() - token_drop_t0

        if create_causal_mask is not None:
            causal_mask = create_causal_mask(
                config=language_model.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=past_key_values,
            )
        else:
            causal_mask = None

        lm_t0 = time.perf_counter()
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=use_cache or False,
                **kwargs,
            )
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                dev = hidden_states.device if not isinstance(hidden_states, tuple) else hidden_states[0].device
                hidden_states = language_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks.to(dev) if visual_pos_masks is not None else None,
                    deepstack_visual_embeds[layer_idx].to(dev),
                )

        hidden_states = language_model.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self._stage_timing["llm_prefill"] += time.perf_counter() - lm_t0

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

        output_cls = Qwen3VLCausalLMOutputWithPast
        if output_cls is None:
            from dataclasses import dataclass
            from transformers.modeling_outputs import ModelOutput
            @dataclass
            class _Out(ModelOutput):
                loss: torch.FloatTensor = None
                logits: torch.FloatTensor = None
                past_key_values = None
                hidden_states = None
                attentions = None
                rope_deltas = None
            output_cls = _Out

        return output_cls(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            rope_deltas=model.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        drop_method=None,
        drop_threshold=None,
        drop_absolute=None,
        dr_save_path=None,
        dp_save_path=None,
        sim_save_path=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        model_inputs["drop_method"] = drop_method
        model_inputs["drop_threshold"] = drop_threshold
        model_inputs["drop_absolute"] = drop_absolute
        model_inputs["dr_save_path"] = dr_save_path
        model_inputs["dp_save_path"] = dp_save_path
        model_inputs["sim_save_path"] = sim_save_path
        return model_inputs
