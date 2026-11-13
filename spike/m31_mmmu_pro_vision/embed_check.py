"""MiniMax-M3.1 image preprocessing and vision embeddings, ATOM against the
transformers reference, on MMMU-Pro vision images.

Preprocessing: the checkpoint's processor, which ATOM's server runs
(AutoProcessor with trust_remote_code), against transformers'
MiniMaxM3VLImageProcessor. Embeddings: ATOM's tower, projector and patch
merger (atom.models.minimax_m3.vl.build_vision_modules) against transformers'
MiniMaxM3VLVisionModel and MiniMaxM3VLMultiModalProjector, both loaded from
the checkpoint's bf16 vision weights and fed the same pixels. The
transformers tower in fp32 is the reference; transformers in bf16 shows the
bf16 rounding floor ATOM's difference is read against.

    python3 embed_check.py [n_images]     (one GPU; prints one JSON line per image, then a summary)
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

MODEL = os.environ.get(
    "MODEL_PATH", "/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4"
)
PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


def vision_weights():
    from safetensors import safe_open

    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    files = {f for name, f in index.items() if name.startswith(PREFIXES)}
    weights = {}
    for f in files:
        with safe_open(f"{MODEL}/{f}", "pt") as st:
            for name in st.keys():
                if name.startswith(PREFIXES):
                    weights[name] = st.get_tensor(name)
    return weights


def atom_modules(weights, device):
    from transformers import AutoConfig

    from atom.models.minimax_m3.vl import build_vision_modules

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tower, projector, merger = build_vision_modules(config)
    modules = {"vision_tower.": tower, "multi_modal_projector.": projector, "patch_merge_mlp.": merger}
    for prefix, module in modules.items():
        params = dict(module.named_parameters())
        for name, tensor in weights.items():
            if not name.startswith(prefix):
                continue
            local = name[len(prefix) :]
            for shard in "qkv":
                if f".{shard}_proj." in local:
                    target = params[local.replace(f".{shard}_proj.", ".qkv_proj.")]
                    target.weight_loader(target, tensor, shard)
                    break
            else:
                params[local].data.copy_(tensor)
        module.to(device=device, dtype=torch.bfloat16).eval()

    def embed(pixels, grid):
        return merger(projector(tower(pixels.to(device), grid.tolist())))

    return embed


def hf_modules(weights, device, dtype):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLMultiModalProjector,
        MiniMaxM3VLVisionModel,
    )

    config = MiniMaxM3VLConfig.from_pretrained(MODEL)
    config.vision_config._attn_implementation = "sdpa"
    tower = MiniMaxM3VLVisionModel(config.vision_config)
    projector = MiniMaxM3VLMultiModalProjector(config)
    renames = [
        ("vision_tower.vision_model.embeddings.patch_embedding.", tower, "embeddings.proj."),
        ("vision_tower.vision_model.encoder.layers.", tower, "layers."),
        ("vision_tower.vision_model.pre_layrnorm.", tower, "pre_layrnorm."),
        ("multi_modal_projector.", projector, ""),
        ("patch_merge_mlp.linear_1.", projector, "merge_linear_1."),
        ("patch_merge_mlp.linear_2.", projector, "merge_linear_2."),
    ]
    loaded = set()
    for name, tensor in weights.items():
        for src, module, dst in renames:
            if name.startswith(src):
                target = dict(module.named_parameters())[dst + name[len(src) :]]
                target.data.copy_(tensor)
                loaded.add(id(target))
                break
    for module in (tower, projector):
        missing = [n for n, p in module.named_parameters() if id(p) not in loaded]
        if missing:
            raise RuntimeError(f"transformers parameters not in the checkpoint: {missing[:5]}")
        module.to(device=device, dtype=dtype).eval()

    def embed(pixels, grid):
        out = tower(pixel_values=pixels.to(device, dtype), grid_thw=grid.to(device))
        return projector(out.last_hidden_state.squeeze(0))

    return embed


def compare(x, ref):
    x, ref = x.float(), ref.float()
    cos = F.cosine_similarity(x, ref, dim=-1)
    return {
        "rel_l2": round(float((x - ref).norm() / ref.norm()), 5),
        "min_token_cos": round(float(cos.min()), 5),
    }


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import (
        MiniMaxM3VLImageProcessor,
    )

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    device = torch.device("cuda")
    served = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True).image_processor
    reference = MiniMaxM3VLImageProcessor.from_pretrained(MODEL)
    weights = vision_weights()
    atom = atom_modules(weights, device)
    hf32 = hf_modules(weights, device, torch.float32)
    hf16 = hf_modules(weights, device, torch.bfloat16)
    data = load_dataset("MMMU/MMMU_Pro", "vision", split="test")
    step = len(data) // n
    rows = []
    with torch.no_grad():
        for i in range(0, step * n, step):
            image = data[i]["image"].convert("RGB")
            a = served(images=[image], return_tensors="pt")
            b = reference(images=[image], return_tensors="pt")
            row = {
                "doc": data[i]["id"],
                "size": list(image.size),
                "grid_served": a["image_grid_thw"][0].tolist(),
                "grid_reference": b["image_grid_thw"][0].tolist(),
            }
            same_grid = row["grid_served"] == row["grid_reference"]
            row["pixels_max_abs_diff"] = (
                float((a["pixel_values"].float() - b["pixel_values"].float()).abs().max())
                if same_grid
                else None
            )
            ref = hf32(b["pixel_values"], b["image_grid_thw"])
            row["atom_vs_hf_fp32"] = compare(atom(a["pixel_values"], a["image_grid_thw"]), ref) if same_grid else None
            row["hf_bf16_vs_hf_fp32"] = compare(hf16(b["pixel_values"], b["image_grid_thw"]), ref)
            rows.append(row)
            print(json.dumps(row), flush=True)
    ok = [r for r in rows if r["atom_vs_hf_fp32"]]
    print(
        json.dumps(
            {
                "images": len(rows),
                "same_grid": len(ok),
                "max_pixels_abs_diff": max((r["pixels_max_abs_diff"] for r in ok), default=None),
                "atom_rel_l2_max": max((r["atom_vs_hf_fp32"]["rel_l2"] for r in ok), default=None),
                "atom_min_token_cos": min((r["atom_vs_hf_fp32"]["min_token_cos"] for r in ok), default=None),
                "hf_bf16_rel_l2_max": max(r["hf_bf16_vs_hf_fp32"]["rel_l2"] for r in rows),
                "hf_bf16_min_token_cos": min(r["hf_bf16_vs_hf_fp32"]["min_token_cos"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
