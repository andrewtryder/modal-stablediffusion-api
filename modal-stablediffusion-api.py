from __future__ import annotations

import io
import os
import time
from pathlib import Path
from modal import Image, Secret, Stub, method, web_endpoint
from fastapi import Response

stub = Stub("stable-diffusion-cli")
model_id = "runwayml/stable-diffusion-v1-5"
cache_path = "/vol/cache"


def download_models():
    import diffusers
    import torch

    hugging_face_token = os.environ["HUGGINGFACE_TOKEN"]

    # Download scheduler configuration. Experiment with different schedulers
    # to identify one that works best for your use-case.
    scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        use_auth_token=hugging_face_token,
        cache_dir=cache_path,
    )
    scheduler.save_pretrained(cache_path, safe_serialization=True)

    # Downloads all other models.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=hugging_face_token,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    )
    pipe.save_pretrained(cache_path, safe_serialization=True)


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torch",
        "torchvision",
        "transformers~=4.25.1",
        "triton",
        "safetensors",
        "torch>=2.0"
    )
    .pip_install("xformers", pre=True)
    .run_function(
        download_models,
        secrets=[Secret.from_name("HUGGINGFACE_TOKEN")],
    )
)
stub.image = image
@stub.cls(gpu="A10G")
class StableDiffusion:
    def __enter__(self):
        import diffusers
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True

        scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            cache_path,
            subfolder="scheduler",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,  # important if steps are <= 10
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            cache_path,
            scheduler=scheduler,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(self, prompt: str, steps: int = 20, batch_size: int = 4) -> bytes:
        import torch

        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    [prompt] * batch_size,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                ).images

        # Convert to PNG bytes
        buf = io.BytesIO()
        try:
            images[0].save(buf, format="PNG")
            return buf.getvalue()
        finally:
            buf.close()

@stub.function()
@web_endpoint(method="GET")
def web(query: str):

    steps = 10
    samples = 1
    batch_size = 1

    dir = Path("stable-diffusion")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    sd = StableDiffusion()
    image = sd.run_inference.call(query, steps, batch_size)
    print("Image")
    return Response(content=image, media_type="image/png")