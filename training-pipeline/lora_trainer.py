"""Built-in LoRA-DreamBooth trainer (text→image adapters) on open diffusion weights.

GPU required. Runs standalone:
    python training-pipeline/lora_trainer.py --dataset datasets/identity --output_dir lora/identity_lora

Emits JSON lines on stdout: {"step": 12, "progress": 0.4, "loss": 0.09, ...}
Saves diffusers-format weights (load with pipe.load_lora_weights(out_dir)).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PIPES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPES / "core-engine"))


def emit(**fields) -> None:
    fields.setdefault("time", round(time.time(), 2))
    print(json.dumps(fields), flush=True)


def emit_progress(step, total, loss) -> None:
    emit(step=step, progress=round(min(1.0, step / max(1, total)), 4), loss=round(float(loss), 6), message="training")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Genesis LoRA identity trainer")
    parser.add_argument("--dataset", required=True, help="folder with resized images and .txt captions")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--trigger", default="genesis_identity")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from PIL import Image

    import torch
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    from genesis.dataset import IMAGE_EXTENSIONS

    torch.manual_seed(args.seed)

    dataset = Path(args.dataset)
    images = sorted(f for f in dataset.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        emit(error="no images found", progress=0.0)
        return 1

    captions: list[str] = []
    for image in images:
        caption_file = image.with_suffix(".txt")
        caption = caption_file.read_text(encoding="utf-8").strip() if caption_file.is_file() else args.trigger
        if args.trigger not in caption:
            caption = f"{args.trigger}, {caption}" if caption else args.trigger
        captions.append(caption)

    emit(message="loading " + args.base, progress=0.0)
    tokenizer = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder")
    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float32)
    unet, vae = pipe.unet, pipe.vae
    noise_scheduler = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet_peft = get_peft_model(unet, lora)
    unet_peft.train()
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    optimizer = torch.optim.AdamW([p for p in unet_peft.parameters() if p.requires_grad], lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        emit(warning="No CUDA GPU; CPU training is only a smoke test.", progress=0.0)
    unet_peft.to(device)
    text_encoder.to(device)
    vae.to(device)

    def encode_batch(batch_paths):
        pixel_values = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB").resize(
                (args.resolution, args.resolution), Image.LANCZOS
            )
            import numpy as np

            pixel_values.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0)
        pixel_values = torch.stack(pixel_values).to(device)
        pixel_values = (pixel_values - 0.5) / 0.5
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * getattr(vae.config, "scaling_factor", 0.18215)
        indices = [images.index(p) for p in batch_paths]
        input_ids = tokenizer([captions[i] for i in indices], padding=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]
        return latents, encoder_hidden_states

    def step_loss(batch_paths):
        latents, encoder_hidden_states = encode_batch(batch_paths)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, int(noise_scheduler.config.num_train_timesteps), (latents.shape[0],), device=device
        )
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        pred = unet_peft(noisy, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        return torch.nn.functional.mse_loss(pred.float(), noise.float(), reduction="mean")

    for step in range(1, args.max_steps + 1):
        start = (step - 1) % len(images)
        batch = images[start : start + args.batch_size] or images[start : start + 1]
        loss = step_loss(batch) / args.gradient_accumulation
        loss.backward()
        if step % args.gradient_accumulation == 0 or step == args.max_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 5 == 0 or step in (1, args.max_steps):
            emit_progress(step, args.max_steps, loss.detach())
        if step % 50 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    state = get_peft_model_state_dict(unet_peft)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.models.lora import convert_state_dict_to_diffusers

        diffusers_state = convert_state_dict_to_diffusers(state)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(out),
            unet_lora_layers=diffusers_state,
            safe_serialization=True,
        )
        emit(message="saved diffusers-format LoRA", progress=1.0)
    except Exception as exc:
        torch.save(state, out / "lora_state.pt")
        emit(warning=f"diffusers save failed ({exc}); raw state saved to lora_state.pt", progress=1.0)

    emit(done=str(out), step=args.max_steps, progress=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
