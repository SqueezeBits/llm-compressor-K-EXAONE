# coding=utf-8
# Copyright 2026 The LG AI Research and llm-compressor contributors.
# Licensed under the Apache License, Version 2.0.
#
# CalibrationExaoneMoeSparseMoEBlock
# ──────────────────────────────────
# Drop-in replacement for ExaoneMoeSparseMoEBlock during calibration.
#
# WHY is_permanent = True?
# ExaoneMoeExperts stores all expert weights as 3D batched tensors:
#
#   gate_up_proj : [num_experts, 2 * intermediate_dim, hidden_dim]
#   down_proj    : [num_experts,     hidden_dim, intermediate_dim]
#
# AWQ / GPTQ quantization hooks target nn.Linear modules, not raw
# Parameter tensors.  We therefore permanently convert ExaoneMoeExperts
# into a ModuleList of individual _ExaoneMoeExpertMLP objects (each with
# real gate_proj / up_proj / down_proj nn.Linear modules) so that the
# observer hooks can attach and collect activation statistics for every
# expert.
#
# CALIBRATION FORWARD
# When calibrate_all_experts=True every token is passed through every
# expert, ensuring that all 128 routed experts see sufficient activation
# statistics.  Only the tokens actually routed to each expert contribute
# to the output (routing weights are applied per the original logic), so
# the numerical output is identical to the original forward pass.

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule


# ─────────────────────────────────────────────────────────────────────────────
# Individual expert MLP (gate + up merged, then down)
# ─────────────────────────────────────────────────────────────────────────────

class _ExaoneMoeExpertMLP(nn.Module):
    """Single routed expert with separate nn.Linear modules.

    Weights are sliced out of the original ExaoneMoeExperts 3D tensors so
    that quantization observers (AWQ / GPTQ) can attach to them normally.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        gate_up_weight: torch.Tensor,  # [2 * intermediate_dim, hidden_dim]
        down_weight: torch.Tensor,     # [hidden_dim, intermediate_dim]
    ):
        super().__init__()
        self.act_fn = act_fn

        # gate and up are packed together in the original implementation; we
        # split them into two separate Linear modules so observers see each
        # projection individually.
        gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)

        # Preserve dtype and device from the source tensors so that the Linear
        # modules land on the same device (e.g. GPU) and in the same dtype
        # (e.g. bfloat16) as the original model — avoiding calibration forward
        # dtype/device mismatches.
        dtype, device = gate_up_weight.dtype, gate_up_weight.device
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype, device=device)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False,
                                   dtype=down_weight.dtype, device=down_weight.device)

        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(down_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Calibration replacement for ExaoneMoeSparseMoEBlock
# ─────────────────────────────────────────────────────────────────────────────

@MoECalibrationModule.register("ExaoneMoeSparseMoEBlock")
class CalibrationExaoneMoeSparseMoEBlock(MoECalibrationModule):
    """Calibration wrapper for ExaoneMoeSparseMoEBlock.

    Replaces the batched ExaoneMoeExperts with a ModuleList of individual
    _ExaoneMoeExpertMLP modules so that quantization observers can attach to
    every expert's linear projections.

    Setting ``is_permanent = True`` means this replacement is kept for the
    entire quantization run (it is not restored after calibration), which is
    required because AWQ / GPTQ compress the individual nn.Linear modules
    in-place.
    """

    is_permanent = True

    def __init__(self, original, config, calibrate_all_experts: bool = True):
        super().__init__()

        self.calibrate_all_experts = calibrate_all_experts
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        # Keep gate router and shared experts as-is (they are plain nn.Linear
        # modules and can be quantized without any special handling).
        self.gate = original.gate
        self.shared_experts = original.shared_experts

        # Build individual expert modules from the batched weight tensors.
        orig_experts = original.experts  # ExaoneMoeExperts
        act_fn = orig_experts.act_fn
        hidden_dim = orig_experts.hidden_dim
        intermediate_dim = orig_experts.intermediate_dim

        self.experts = nn.ModuleList(
            [
                _ExaoneMoeExpertMLP(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    act_fn=act_fn,
                    gate_up_weight=orig_experts.gate_up_proj[i].detach(),
                    down_weight=orig_experts.down_proj[i].detach(),
                )
                for i in range(self.n_routed_experts)
            ]
        )

    # ------------------------------------------------------------------
    # Routing — mirrors ExaoneMoeSparseMoEBlock.route_tokens_to_experts()
    # ------------------------------------------------------------------

    def route_tokens_to_experts(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        group_scores = (
            router_logits_for_choice
            .view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # ── Routing ───────────────────────────────────────────────────
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # expert_mask: [num_experts, top_k, num_tokens]
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts).permute(2, 1, 0)

        final_hidden_states = torch.zeros_like(hidden_states_flat)

        for expert_idx, expert_layer in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # Pass ALL tokens through the expert so every expert sees
                # representative activations; only routed tokens contribute
                # to the output.
                expert_out = expert_layer(hidden_states_flat)[token_idx]
            else:
                # Standard routing: only process tokens assigned to this expert.
                if len(token_idx) == 0:
                    continue
                expert_out = expert_layer(hidden_states_flat[token_idx])

            if len(token_idx) > 0:
                weighted = expert_out * topk_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))

        routed_out = final_hidden_states.view(*orig_shape)

        # ── Shared experts (always active for every token) ────────────
        output = routed_out + self.shared_experts(residuals)
        return output

    # ------------------------------------------------------------------
    # Restore — not called (is_permanent = True) but required by ABC
    # ------------------------------------------------------------------

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
