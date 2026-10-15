"""Tiny synthetic model/optimizer/EMA/reload test; never a scientific benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path


def run_smoke(output_dir: Path, device_name: str) -> dict[str, object]:
    from .safety import check_gpu_launch_policy, require_single_process
    require_single_process()
    if device_name == "cuda":
        check_gpu_launch_policy()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from pbisc_sdd.diffusion import SparseDataDiffusion, SparseDiffusionConfig
    from pbisc_sdd.ema import ExponentialMovingAverage
    from pbisc_sdd.model import ConditionalSBDDenoiser, SparseDenoiserConfig

    if device_name == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA BF16 is required for the H100 smoke test")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    torch.set_num_threads(1)
    torch.manual_seed(20260627)
    n_genes = 32
    model_config = SparseDenoiserConfig(n_genes=n_genes, condition_dim=3 * n_genes + 6,
                                       hidden_dims=(32, 32, 16, 8), context_dim=16,
                                       time_embedding_dim=16)
    diffusion_config = SparseDiffusionConfig(center_values=True, value_loss_mode="all_measured")

    def create():
        model = ConditionalSBDDenoiser(model_config).to(device)
        diffusion = SparseDataDiffusion(model, diffusion_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        ema = ExponentialMovingAverage(model, decay=0.999)
        return model, diffusion, optimizer, ema

    model, diffusion, optimizer, ema = create()
    mask = torch.ones(8, n_genes, device=device)
    mask[:, -3:] = 0
    values = torch.rand(8, n_genes, device=device) * mask
    values[values < 0.4] = 0
    condition = torch.randn(8, 3 * n_genes + 6, device=device)

    def step(net, optim, average) -> float:
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss = net(values, condition, measurement_mask=mask).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.denoiser.parameters(), 5.0)
        optim.step()
        average.update(net.denoiser)
        return float(loss.detach().cpu())

    first = step(diffusion, optimizer, ema)
    checkpoint = output_dir / "synthetic_checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}, checkpoint)
    second = step(diffusion, optimizer, ema)
    expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_ema = {key: value.detach().clone() for key, value in ema.model.state_dict().items()}
    resumed_model, resumed_diffusion, resumed_optimizer, resumed_ema = create()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed_model.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_ema.load_state_dict(state["ema"])
    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"])
    replayed = step(resumed_diffusion, resumed_optimizer, resumed_ema)
    exact_model = all(torch.equal(value, resumed_model.state_dict()[key])
                      for key, value in expected_model.items())
    exact_ema = all(torch.equal(value, resumed_ema.model.state_dict()[key])
                    for key, value in expected_ema.items())
    if not exact_model or not exact_ema or second != replayed:
        raise RuntimeError("synthetic optimizer/RNG/EMA continuation diverged")
    resumed_model.eval()
    generated = resumed_diffusion.sample(condition, measurement_mask=mask, steps=4)
    unavailable_zero = bool((generated.values[mask == 0] == 0).all())
    finite = bool(torch.isfinite(generated.values).all())
    if not unavailable_zero or not finite:
        raise RuntimeError("synthetic sampler violated finite/measurement-mask contract")
    report = {"status": "PASS", "test": "synthetic_model_contract_only",
              "full_corpus_training_tested": False, "device": str(device),
              "torch": torch.__version__, "amp": "bf16" if device.type == "cuda" else "none",
              "model_config": asdict(model_config), "optimizer_steps": 2,
              "losses": [first, second], "exact_resumed_model": exact_model,
              "exact_resumed_ema": exact_ema, "unavailable_output_zero": unavailable_zero,
              "generated_finite": finite, "cuda_initialized": torch.cuda.is_initialized(),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (output_dir / "SMOKE_RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(args.output_dir, args.device), indent=2))
    return 0
