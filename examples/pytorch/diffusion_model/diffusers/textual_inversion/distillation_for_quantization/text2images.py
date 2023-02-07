
import argparse
import math
import os
import torch
from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from neural_compressor.utils.pytorch import load
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "-m",
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "-c",
        "--caption",
        type=str,
        default="robotic cat with wings",
        help="Text used to generate images.",
    )
    parser.add_argument(
        "-n",
        "--images_num",
        type=int,
        default=4,
        help="How much images to generate.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for random process.",
    )
    parser.add_argument(
        "-ci",
        "--cuda_id",
        type=int,
        default=0,
        help="cuda_id.",
    )
    parser.add_argument(
        "--bf16",
        dest='bf16',
        action='store_true',
        help="use ipex bf16.",
    )
    parser.add_argument(
        "--int8",
        dest='int8',
        action='store_true',
        help="use int8.",
    )
    parser.add_argument(
        "--jit",
        dest='jit',
        action='store_true',
        help="use jit trace.",
    )

    args = parser.parse_args()
    return args


def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def generate_images(
    pipeline,
    prompt="robotic cat with wings",
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    seed=42
):
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    images = pipeline(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    _rows = int(math.sqrt(num_images_per_prompt))
    grid = image_grid(images, rows=_rows, cols=num_images_per_prompt // _rows)
    return grid, images

args = parse_args()
# Load models and create wrapper for stable diffusion
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

pipeline = StableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=PNDMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler"),
    safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
    feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
)
pipeline.safety_checker = lambda images, clip_input: (images, False)
if args.int8:
    assert os.path.exists(os.path.join(args.pretrained_model_name_or_path, "best_model.pt"))
    unet = load(args.pretrained_model_name_or_path, model=unet)
    unet.eval()
    if args.jit:
        sample = torch.randn(2, 4, 64, 64)
        timestep = torch.rand(1)*999
        encoder_hidden_status = torch.randn(2, 77, 768)
        input_example = (sample, timestep, encoder_hidden_status)
        unet = torch.jit.trace(unet, input_example, strict=False)
    setattr(pipeline, "unet", unet)
else:
    unet = unet.to(torch.device("cuda", args.cuda_id))

if args.bf16:
    import intel_extension_for_pytorch as ipex
    pipeline.vae = ipex.optimize(pipeline.vae.eval(), dtype=torch.bfloat16, inplace=True)
    pipeline.safety_checker = ipex.optimize(pipeline.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)

pipeline = pipeline.to(unet.device)
grid, images = generate_images(pipeline, prompt=args.caption, num_images_per_prompt=args.images_num, seed=args.seed)
grid.save(
    os.path.join(args.pretrained_model_name_or_path, "{}.png".format("_".join(args.caption.split())))
)
dirname = os.path.join(args.pretrained_model_name_or_path, "_".join(args.caption.split()))
os.makedirs(dirname, exist_ok=True)
for idx, image in enumerate(images):
    image.save(os.path.join(dirname, "{}.png".format(idx+1)))