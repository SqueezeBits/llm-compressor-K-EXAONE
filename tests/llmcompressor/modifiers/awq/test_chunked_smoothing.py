"""
Regression / unit tests for AWQModifier.cache_chunk_size_batches (chunked smoothing).

Covers:
  1. Field is accepted by the pydantic model and defaults to None.
  2. When chunking is enabled, _setup_activation_cache_hooks does NOT register
     the parent-kwargs hook (so _parent_args_cache stays empty after calibration).
  3. _compute_chunk_loss_sum gives unnormalised sums that accumulate to the same
     result as _compute_loss (which is normalised over all batches).
  4. End-to-end: chunked and non-chunked produce weights that are numerically
     close (same best-candidate selection, same smoothing applied).
"""

import copy
import inspect
from itertools import product
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor.modifiers.awq import AWQMapping, AWQModifier


# ---------------------------------------------------------------------------
# 1. Field validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cache_chunk_size_batches_field():
    """cache_chunk_size_batches must default to None and accept int."""
    m = AWQModifier(scheme="W4A16")
    assert m.cache_chunk_size_batches is None

    m2 = AWQModifier(scheme="W4A16", cache_chunk_size_batches=8)
    assert m2.cache_chunk_size_batches == 8


# ---------------------------------------------------------------------------
# 2. Hook registration: parent kwargs hook skipped when chunking enabled
# ---------------------------------------------------------------------------


def _make_tiny_model():
    """Return a trivial 2-layer linear model with a LN smooth layer."""
    ln = nn.LayerNorm(8)
    fc1 = nn.Linear(8, 8, bias=False)
    fc2 = nn.Linear(8, 8, bias=False)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = ln
            self.fc1 = fc1
            self.fc2 = fc2

        def forward(self, x):
            h = self.ln(x)
            return self.fc1(h) + self.fc2(h)

    model = _Block()
    return model, ln, fc1, fc2


@pytest.mark.unit
def test_no_parent_kwarg_hook_in_chunked_mode():
    """In chunked mode _parent_args_cache should exist but accumulate no data."""
    model, ln, fc1, fc2 = _make_tiny_model()

    awq = AWQModifier(
        mappings=[AWQMapping("ln", ["fc1", "fc2"])],
        scheme="W4A16",
        cache_chunk_size_batches=4,
        offload_device=None,
    )

    # Manually resolve mappings (skips pydantic-based initialization)
    from llmcompressor.modifiers.awq.base import get_lowest_common_ancestor_with_avoid
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping

    parent_name, parent = get_lowest_common_ancestor_with_avoid(["fc1", "fc2"], model)
    awq._resolved_mappings = [
        ResolvedMapping(
            smooth_name="ln",
            smooth_layer=ln,
            balance_layers=[fc1, fc2],
            balance_names=["fc1", "fc2"],
            parent=parent,
            parent_name=parent_name,
            activation_hook_target=None,
        )
    ]

    awq._setup_activation_cache_hooks()

    # The parent cache dict must contain an entry (for deduplication), but
    # the entry must be EMPTY (no batches were ever cached).
    assert parent in awq._parent_args_cache
    assert len(awq._parent_args_cache[parent]) == 0

    # Simulate a forward pass — the parent-kwargs hook must NOT fire.
    x = torch.randn(2, 8)
    model(x)
    assert len(awq._parent_args_cache[parent]) == 0, (
        "Parent kwargs accumulated in chunked mode — hook should NOT be registered"
    )

    awq.remove_hooks()


@pytest.mark.unit
def test_parent_kwarg_hook_in_non_chunked_mode():
    """In non-chunked mode the parent-kwargs hook IS registered and accumulates."""
    model, ln, fc1, fc2 = _make_tiny_model()

    awq = AWQModifier(
        mappings=[AWQMapping("ln", ["fc1", "fc2"])],
        scheme="W4A16",
        cache_chunk_size_batches=None,  # non-chunked
        offload_device=None,
    )

    from llmcompressor.modifiers.awq.base import get_lowest_common_ancestor_with_avoid
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping

    parent_name, parent = get_lowest_common_ancestor_with_avoid(["fc1", "fc2"], model)
    awq._resolved_mappings = [
        ResolvedMapping(
            smooth_name="ln",
            smooth_layer=ln,
            balance_layers=[fc1, fc2],
            balance_names=["fc1", "fc2"],
            parent=parent,
            parent_name=parent_name,
            activation_hook_target=None,
        )
    ]

    awq._setup_activation_cache_hooks()

    # Forward pass — parent-kwargs hook should fire and cache one entry.
    x = torch.randn(2, 8)
    model(x)
    assert len(awq._parent_args_cache[parent]) == 1, (
        "Parent kwargs should have been captured in non-chunked mode"
    )

    awq.remove_hooks()


# ---------------------------------------------------------------------------
# 3. _compute_chunk_loss_sum sums correctly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_chunk_loss_sum_matches_compute_loss():
    """
    Calling _compute_chunk_loss_sum on each chunk and summing must give the
    same total as _compute_loss over all batches (after normalisation).
    """
    torch.manual_seed(0)
    n_batches = 6
    chunk_size = 2
    hidden = 8

    fp16_outputs = [torch.randn(2, hidden) for _ in range(n_batches)]
    int_w_outputs = [torch.randn(2, hidden) for _ in range(n_batches)]

    awq = AWQModifier(scheme="W4A16")

    # Mock session state with no loss masks
    mock_state = MagicMock()
    mock_state.loss_masks = None
    mock_session = MagicMock()
    mock_session.state = mock_state

    with patch(
        "llmcompressor.modifiers.awq.base.active_session", return_value=mock_session
    ):
        # Chunked accumulation
        total_loss_sum = 0.0
        total_n_elem = 0
        for chunk_start in range(0, n_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_batches)
            ls, ne = awq._compute_chunk_loss_sum(
                fp16_outputs[chunk_start:chunk_end],
                int_w_outputs[chunk_start:chunk_end],
                chunk_start,
            )
            total_loss_sum += ls
            total_n_elem += ne

        chunked_loss = total_loss_sum / total_n_elem

        # Direct reference: compute MSE over all batches
        ref_loss = 0.0
        ref_n = 0
        for fp16, intw in zip(fp16_outputs, int_w_outputs):
            ref_loss += F.mse_loss(fp16, intw, reduction="sum").item()
            ref_n += fp16.numel()
        ref_loss /= ref_n

    assert abs(chunked_loss - ref_loss) < 1e-5, (
        f"Chunked loss {chunked_loss:.6f} differs from reference {ref_loss:.6f}"
    )


# ---------------------------------------------------------------------------
# 4. End-to-end: chunked vs non-chunked produce equivalent smoothed weights
# ---------------------------------------------------------------------------


class _TinyTransformerBlock(nn.Module):
    """Minimal block: LayerNorm + two parallel Linear projections."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        h = self.ln(x)
        return self.fc1(h) + self.fc2(h)


def _run_awq_scale_search(
    model_ref: nn.Module,
    chunk_size: int | None,
    n_batches: int = 8,
    batch_dim: int = 4,
    hidden: int = 16,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Directly exercise AWQModifier._compute_best_scale (non-chunked) and
    _apply_smoothing_chunked (chunked) on identical synthetic data,
    returning the fc1/fc2 weights after smoothing.

    This test does NOT run the full sequential pipeline.  It manually sets up
    the modifier state (resolved mappings, activation stats, parent-args cache)
    and calls the relevant smoothing methods.
    """
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
    )
    from compressed_tensors.quantization.lifecycle.initialize import (
        initialize_module_for_quantization,
    )

    from llmcompressor.modifiers.awq.base import get_lowest_common_ancestor_with_avoid
    from llmcompressor.modifiers.awq.mappings import ResolvedMapping
    from llmcompressor.pipelines.cache import IntermediatesCache

    torch.manual_seed(seed)
    model = copy.deepcopy(model_ref)

    # Attach a minimal quantization scheme to fc1 and fc2
    q_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=False,
            strategy=QuantizationStrategy.GROUP,
            group_size=8,
        ),
    )
    for module in [model.fc1, model.fc2]:
        initialize_module_for_quantization(module, q_scheme)

    parent_name, parent = get_lowest_common_ancestor_with_avoid(
        ["fc1", "fc2"], model
    )

    awq = AWQModifier(
        mappings=[AWQMapping("ln", ["fc1", "fc2"])],
        scheme="W4A16",
        duo_scaling=True,
        n_grid=4,  # small for speed
        offload_device=None,
        cache_chunk_size_batches=chunk_size,
    )
    awq._resolved_mappings = [
        ResolvedMapping(
            smooth_name="ln",
            smooth_layer=model.ln,
            balance_layers=[model.fc1, model.fc2],
            balance_names=["fc1", "fc2"],
            parent=parent,
            parent_name=parent_name,
            activation_hook_target=None,
        )
    ]

    # Synthetic calibration batches (same seed → identical data for both runs)
    torch.manual_seed(seed + 1)
    batches = [torch.randn(batch_dim, hidden) for _ in range(n_batches)]

    # ── Accumulate activation stats (mimics the calibration-pass hook) ─────────
    x_sum = torch.zeros(hidden)
    x_count = torch.tensor(0)
    for x in batches:
        h = model.ln(x)
        inp = h.abs().flatten(0, -2)
        x_sum += inp.float().sum(0)
        x_count += inp.size(0)
    awq._smooth_activation_stats["ln"] = [x_sum, x_count]

    # ── Mock active_session ──────────────────────────────────────────────────
    mock_state = MagicMock()
    mock_state.loss_masks = None
    mock_state.current_batch_idx = 0
    mock_state.sequential_prefetch = False
    mock_session = MagicMock()
    mock_session.state = mock_state

    from compressed_tensors.quantization import disable_quantization

    model.apply(disable_quantization)

    if chunk_size is None:
        # Non-chunked: fill _parent_args_cache manually and call _apply_smoothing
        awq._parent_args_cache[parent] = IntermediatesCache(None, None)
        for x in batches:
            vals = inspect.signature(parent.forward).bind(x)
            awq._parent_args_cache[parent].append(vals.arguments)

        with patch(
            "llmcompressor.modifiers.awq.base.active_session",
            return_value=mock_session,
        ):
            with torch.no_grad():
                awq._apply_smoothing(model)
    else:
        # Chunked: build a fake IntermediatesCache and a fake subgraph,
        # then call _apply_smoothing_chunked.
        fake_cache = IntermediatesCache(None, None)
        for x in batches:
            fake_cache.append({"x": x})

        class _FakeSubgraph:
            input_names = {"x"}

            @staticmethod
            def forward(mdl, x):
                return mdl(x)

        mock_state.current_activations = fake_cache
        mock_state.current_subgraph = _FakeSubgraph()

        with patch(
            "llmcompressor.modifiers.awq.base.active_session",
            return_value=mock_session,
        ):
            with torch.no_grad():
                awq._apply_smoothing_chunked(model, mock_state)

    return {
        "fc1_weight": model.fc1.weight.detach().clone(),
        "fc2_weight": model.fc2.weight.detach().clone(),
        "ln_weight": model.ln.weight.detach().clone(),
    }


@pytest.mark.unit
def test_chunked_vs_non_chunked_equivalent():
    """
    Chunked and non-chunked AWQ smoothing should yield weights that are
    numerically close (within floating-point accumulation tolerance).
    """
    model_ref = _TinyTransformerBlock(dim=16)

    ref_weights = _run_awq_scale_search(model_ref, chunk_size=None)
    chunked_weights = _run_awq_scale_search(model_ref, chunk_size=4)

    for key in ref_weights:
        torch.testing.assert_close(
            ref_weights[key],
            chunked_weights[key],
            atol=1e-4,
            rtol=1e-4,
            msg=f"Mismatch in {key}",
        )


@pytest.mark.unit
def test_chunk_size_1_equivalent():
    """chunk_size=1 (extreme case) should still match non-chunked."""
    model_ref = _TinyTransformerBlock(dim=16)
    ref_weights = _run_awq_scale_search(model_ref, chunk_size=None)
    chunked_weights = _run_awq_scale_search(model_ref, chunk_size=1)

    for key in ref_weights:
        torch.testing.assert_close(
            ref_weights[key],
            chunked_weights[key],
            atol=1e-4,
            rtol=1e-4,
            msg=f"Mismatch in {key} (chunk_size=1)",
        )


@pytest.mark.unit
def test_chunk_size_larger_than_dataset():
    """chunk_size >= n_batches should behave identically to non-chunked."""
    model_ref = _TinyTransformerBlock(dim=16)
    n_batches = 6
    ref_weights = _run_awq_scale_search(
        model_ref, chunk_size=None, n_batches=n_batches
    )
    chunked_weights = _run_awq_scale_search(
        model_ref, chunk_size=100, n_batches=n_batches
    )

    for key in ref_weights:
        torch.testing.assert_close(
            ref_weights[key],
            chunked_weights[key],
            atol=1e-4,
            rtol=1e-4,
            msg=f"Mismatch in {key} (chunk_size > n_batches)",
        )
