"""
Single-layer attention collector for Text-to-Video (T2V) attention extraction.

Hooks into a *single* user-specified LM decoder layer to collect the attention
map during the prefill forward pass, then aggregates per-patch and per-frame
weights using the same convention as the original ``AttentionCollector``:

- Mean over all attention heads.
- For each video token: max attention weight across all text query positions.
- For each video frame: mean of its constituent video-token weights.
"""

import torch
import torch.nn as nn


class LayerAttentionCollector:
    """Collect T2V attention from a single LM self-attention layer.

    Parameters
    ----------
    layer_idx : int
        Which LM decoder layer to hook (supports negative indexing).
    """

    def __init__(self, layer_idx: int = -1):
        self.layer_idx = layer_idx
        self._hooks = []
        self.reset()

    def reset(self):
        self.text_indices = None
        self.frame_info = []
        self.attn_weights = None  # raw (seq, seq) after head-mean

    # ------------------------------------------------------------------
    # Frame / text token parsing (two paths, same as original collector)
    # ------------------------------------------------------------------

    def _parse_from_hook(self, visual_pos_masks, position_ids):
        """Parse frame info from model-injected ``visual_pos_masks``."""
        vis_mask = visual_pos_masks[0]
        pos = position_ids[:, 0]
        vis_idx = vis_mask.nonzero(as_tuple=True)[0]
        self.text_indices = (~vis_mask).nonzero(as_tuple=True)[0].cpu()
        if vis_idx.numel() == 0:
            return
        t_pos = pos[0, vis_idx]
        for t_val in t_pos.unique(sorted=True):
            sel = (t_pos == t_val).nonzero(as_tuple=True)[0]
            fvi = vis_idx[sel]
            h_ids = pos[1, fvi]
            w_ids = pos[2, fvi]
            h0, w0 = h_ids.min().item(), w_ids.min().item()
            rh = (h_ids - h0).tolist()
            rw = (w_ids - w0).tolist()
            self.frame_info.append({
                'vis_indices': fvi.cpu(),
                'grid_h': max(rh) + 1,
                'grid_w': max(rw) + 1,
            })

    def parse_from_input_ids(self, input_ids, video_grid_thw, config):
        """Parse frame info directly from ``input_ids`` and ``video_grid_thw``."""
        ids = input_ids[0]
        vis_s = config.vision_start_token_id
        vis_e = config.vision_end_token_id
        vid_id = config.video_token_id
        starts = (ids == vis_s).nonzero(as_tuple=True)[0]
        ends = (ids == vis_e).nonzero(as_tuple=True)[0]
        segs = []
        for s, e in zip(starts, ends):
            if ids[s + 1] == vid_id:
                segs.append((s.item() + 1, e.item()))
        all_vis = set()
        for s, e in segs:
            all_vis.update(range(s, e))
        self.text_indices = torch.tensor(
            sorted(set(range(len(ids))) - all_vis), dtype=torch.long)
        T, H, W = video_grid_thw[0].tolist()
        tpf = H * W
        for seg_s, seg_e in segs:
            n_f = (seg_e - seg_s) // tpf
            for f in range(n_f):
                base = seg_s + f * tpf
                self.frame_info.append({
                    'vis_indices': torch.arange(base, base + tpf),
                    'grid_h': H,
                    'grid_w': W,
                })

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _lm_pre_hook(self, module, args, kwargs):
        vpm = kwargs.get('visual_pos_masks')
        pid = kwargs.get('position_ids')
        if vpm is not None and pid is not None:
            self.frame_info = []
            self._parse_from_hook(vpm, pid)

    def _attn_pre_hook(self, module, args, kwargs):
        """Force attention output only for the hooked layer.

        This avoids enabling ``output_attentions=True`` globally,
        which can explode memory on long video sequences.
        """
        kw = dict(kwargs) if kwargs is not None else {}
        kw['output_attentions'] = True
        return args, kw

    def _attn_post_hook(self, module, input, output):
        if not self.frame_info or self.text_indices is None:
            return output
        aw = output[1]
        if aw is None:
            return output
        self.attn_weights = aw[0].float().mean(0).detach().cpu()  # (seq, seq)
        out = list(output)
        out[1] = None
        return tuple(out)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model):
        """Register hooks on the model.

        Model hierarchy: model.model.language_model.layers[i].self_attn
        """
        lm = model.model.language_model
        self._hooks.append(
            lm.register_forward_pre_hook(self._lm_pre_hook, with_kwargs=True))

        num_layers = len(lm.layers)
        idx = self.layer_idx
        if idx < 0:
            idx = num_layers + idx
        if not (0 <= idx < num_layers):
            raise ValueError(
                f"layer_idx={self.layer_idx} out of range for model with "
                f"{num_layers} layers (resolved to {idx})")
        target = lm.layers[idx].self_attn
        self._hooks.append(
            target.register_forward_pre_hook(self._attn_pre_hook, with_kwargs=True))
        self._hooks.append(
            target.register_forward_hook(self._attn_post_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_frame_weights(self):
        """Compute per-frame weights from the captured attention map.

        Returns
        -------
        list[dict]
            Each dict has keys: ``frame_index``, ``frame_weight``,
            ``patch_weights``, ``num_patches``.
        """
        if self.attn_weights is None or not self.frame_info:
            return []

        attn = self.attn_weights
        dev = attn.device
        ti = self.text_indices.to(dev)

        results = []
        for k, info in enumerate(self.frame_info):
            vi = info['vis_indices'].to(dev)
            t2v = attn[ti][:, vi]                       # (n_text, n_patches)
            patch_w = t2v.max(dim=0).values.detach().cpu()  # text-dim max
            frame_w = float(patch_w.mean().item())          # patch mean
            results.append({
                'frame_index': k,
                'num_patches': int(patch_w.numel()),
                'patch_weights': [round(v, 8) for v in patch_w.tolist()],
                'frame_weight': frame_w,
            })
        return results
