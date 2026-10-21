"""Crazy-features: style presets, adapter stacking, prompt enrichment."""
from __future__ import annotations

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "cinematic": {
        "prefix": "cinematic film still, anamorphic lens, dramatic lighting,",
        "suffix": "shot on 35mm, shallow depth of field, high detail",
        "negative": "cartoon, flat lighting, low quality, watermark",
    },
    "anime": {
        "prefix": "modern anime key visual, vivid cel shading,",
        "suffix": "crisp lineart, expressive eyes, studio quality",
        "negative": "photorealistic, 3d render, blurry, watermark",
    },
    "3d": {
        "prefix": "3d render, pixar-style subsurface skin, soft studio HDRI,",
        "suffix": "octane render, ray traced shadows, 8k",
        "negative": "flat 2d, sketch, jpeg artifacts, watermark",
    },
    "noir": {
        "prefix": "black and white film noir, hard key light, venetian blind shadows,",
        "suffix": "grainy 1940s photography, high contrast",
        "negative": "color, pastel, low contrast, watermark",
    },
    "documentary": {
        "prefix": "raw documentary photography, natural light,",
        "suffix": "photojournalism, candid framing, realistic skin texture",
        "negative": "CGI, glossy studio look, watermark",
    },
}


def apply_style(prompt: str, style: str | None, negative_prompt: str = "") -> tuple[str, str]:
    """Blend a style preset into prompt+negative (identity stays first)."""
    preset = STYLE_PRESETS.get((style or "").lower())
    if not preset:
        return prompt, negative_prompt
    styled = f"{preset['prefix']} {prompt.strip()}, {preset['suffix']}".replace(",,", ",")
    negatives = " ".join(x for x in (negative_prompt, preset["negative"]) if x)
    return styled, negatives


def merge_adapters(pipe, adapters: list[str], scales: list[float] | None = None, adapter_name: str = "genesis") -> dict:
    """Stack several LoRAs (identity + style + motion) into one fused adapter."""
    if not adapters:
        return {"fused": False, "names": []}
    weights = scales if scales else [1.0] * len(adapters)
    for path, weight in zip(adapters, weights):
        pipe.load_lora_weights(path, adapter_name=adapter_name)
        pipe.set_adapters(adapter_name, adapter_weights=[float(weight)])
        adapter_name = adapter_name + "-" + str(len(adapters))
    if hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora()
    return {"fused": True, "names": adapters}
