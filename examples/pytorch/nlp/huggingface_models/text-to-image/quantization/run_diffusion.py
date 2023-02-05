#!/usr/bin/env python
# coding=utf-8
#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" Example for stable-diffusion to generate a picture from a text ."""
# You can also adapt this script on your own text to image task. Pointers for this are left as comments.

import argparse
import logging
import inspect
import math
import numpy as np
import os
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List, Optional, Union

from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from pytorch_fid import fid_score


os.environ["CUDA_VISIBLE_DEVICES"] = ""


logger = logging.getLogger(__name__)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Example of a post-training quantization script.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="a drawing of a gray and black dragon",
        help="The input of the model, like: 'a photo of an astronaut riding a horse on mars'.",
    )
    parser.add_argument(
        "--calib_text",
        type=str,
        default="Womens Princess Little Deer Native American Costume",
        help="The calibration data of the model, like: 'Womens Princess Little Deer Native American Costume'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_results",
        help="The path to save model and quantization configures.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt, defaults to 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="random seed",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed: local_rank",
    )
    parser.add_argument(
        "--base_images",
        type=str,
        default="base_images",
        help="Path to training images for FID input.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Whether or not to apply quantization.",
    )
    parser.add_argument(
        "--quantization_approach",
        type=str,
        default="static",
        help="Quantization approach. Supported approach are static, "
                  "dynamic and auto.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        help="Deep learning framework. Supported framework are pytorch, ipex",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="eval_f1",
        help="Metric used for the tuning strategy.",
    )
    parser.add_argument(
        "--is_relative",
        type=bool,
        default="True",
        help="Metric tolerance mode, True for relative, otherwise for absolute.",
    )
    parser.add_argument(
        "--perf_tol",
        type=float,
        default=0.01,
        help="Performance tolerance when optimizing the model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Only test performance for model.",
    )
    parser.add_argument(
        "--accuracy_only",
        action="store_true",
        help="Only test accuracy for model.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="benchmark for int8 model",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="benchmark for bf16 model",
    )
    parser.add_argument(
        "--ipex",
        action="store_true",
        help="benchmark with IPEX",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

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


def benchmark(pipe, generator):
    warmup = 2
    total = 5
    total_time = 0
    with torch.no_grad():
        for i in range(total):
            prompt = "a photo of an astronaut riding a horse on mars"
            start2 = time.time()
            images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
            end2 = time.time()
            if i >= warmup:
                total_time += end2 - start2
    print("Average Latency: ", (total_time) / (total - warmup), "s")
    print("Average Throughput: {:.5f} samples/sec".format((total - warmup) / (total_time)))


def accuracy(pipe, generator, rows, args):
    with torch.no_grad():
        if args.bf16:
            print("---- enable AMP bf16 model ----")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                new_images = pipe(
                    args.input_text,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    generator=generator,
                    num_images_per_prompt=args.num_images_per_prompt,
                ).images
        else:
            new_images = pipe(
                args.input_text,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=generator,
                num_images_per_prompt=args.num_images_per_prompt,
            ).images
        tmp_save_images = "tmp_save_images"
        os.makedirs(tmp_save_images, exist_ok=True)
        if os.path.isfile(os.path.join(tmp_save_images, "image.png")):
            os.remove(os.path.join(tmp_save_images, "image.png"))
        grid = image_grid(new_images, rows=rows, cols=args.num_images_per_prompt // rows)
        grid.save(os.path.join(tmp_save_images, "image.png"))
        fid = fid_score.calculate_fid_given_paths((args.base_images, tmp_save_images), 1, "cpu", 2048, 8)
        print("Finally FID score Accuracy: {}".format(fid))
        return fid

class CalibDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = "a photo of an astronaut riding a horse on mars"
        return data


def main():
    # Passing the --help flag to this script.

    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Parameters {args}")

    # Set seed before initializing model.
    set_seed(args.seed)

    # Load pretrained model and generate a pipeline
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name_or_path)
    _rows = int(math.sqrt(args.num_images_per_prompt))

    class CalibDataloader:
        def __init__(self):
            self.batch_size = 1

            self.prompts = ["a photo of an astronaut riding a horse on mars",
                            ##"a photo of an astronaut riding a bike on mars",
                            ]
            global my_prompts
            import random
            random.shuffle(my_prompts)
            self.prompts = my_prompts[0:20]
            global my_captions
            self.prompts.extend(my_captions)
            # self.prompts = my_captions
            from diffusers import StableDiffusionOnnxPipeline

            # self.pipe = StableDiffusionOnnxPipeline.from_pretrained(
            #     "runwayml/stable-diffusion-v1-5",
            #     revision="onnx",
            #     provider="CPUExecutionProvider",
            # )

            self.unet_inputs = []
            for prompt in self.prompts:
                self.get_unet_input(prompt)

        def __iter__(self):
            for data in self.unet_inputs:
                yield data

        def get_unet_input(
                self,
                prompt: Union[str, List[str]],
                height: Optional[int] = 512,
                width: Optional[int] = 512,
                num_inference_steps: Optional[int] = 50,
                guidance_scale: Optional[float] = 7.5,
                negative_prompt: Optional[Union[str, List[str]]] = None,
                num_images_per_prompt: Optional[int] = 1,
                eta: Optional[float] = 0.0,
                generator: Optional[np.random.RandomState] = None,
                latents: Optional[np.ndarray] = None,
                output_type: Optional[str] = "pil",
                return_dict: bool = True,
                callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
                callback_steps: Optional[int] = 1,
        ):
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            if (callback_steps is None) or (
                    callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
            ):
                raise ValueError(
                    f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                    f" {type(callback_steps)}."
                )

            if generator is None:
                generator = np.random

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            text_embeddings = self.pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # get the initial random noise unless the user supplied it
            latents_dtype = text_embeddings.dtype
            latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
            if latents is None:
                latents = generator.randn(*latents_shape).astype(latents_dtype)
            elif latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

            # set timesteps
            self.pipe.scheduler.set_timesteps(num_inference_steps)

            latents = latents * np.float(self.pipe.scheduler.init_noise_sigma)

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.pipe.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            # timestep_dtype = next(
            #     (input.type for input in self.pipe.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
            # )
            timestep_dtype = np.int64

            for i, t in enumerate(self.pipe.progress_bar(self.pipe.scheduler.timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.pipe.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
                latent_model_input = latent_model_input.cpu().numpy()

                # predict the noise residual
                timestep = np.array([t], dtype=timestep_dtype)
                if i in range(5, 45):
                    self.unet_inputs.append(((latent_model_input, timestep, text_embeddings), 0))
                else:
                    pass
                noise_pred = self.pipe.unet(sample=latent_model_input, timestep=timestep,
                                            encoder_hidden_states=text_embeddings)
                noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.pipe.scheduler.step(
                    torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
                )
                latents = scheduler_output.prev_sample.numpy()

                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

                ##noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=text_embeddings)

    if args.ipex:
        import intel_extension_for_pytorch as ipex

    if args.tune:
        tmp_fp32_images = "tmp_fp32_images"
        tmp_int8_images = "tmp_int8_images"
        os.makedirs(tmp_fp32_images, exist_ok=True)
        os.makedirs(tmp_int8_images, exist_ok=True)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        fp32_images = pipe(
            args.input_text,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images
        grid = image_grid(fp32_images, rows=_rows, cols=args.num_images_per_prompt // _rows)
        grid.save(os.path.join(tmp_fp32_images, "fp32.png"))

        attr_list = ["unet"]
        for name in attr_list:
            model = getattr(pipe, name)

            def calibration_func(model):
                calib_num = 5
                setattr(pipe, name, model)
                with torch.no_grad():
                    for i in range(calib_num):
                        pipe(
                            args.calib_text,
                            guidance_scale=7.5,
                            num_inference_steps=50,
                            generator=generator,
                            num_images_per_prompt=args.num_images_per_prompt,
                        )

            def eval_func(model):
                setattr(pipe, name, model)
                generator = torch.Generator("cpu").manual_seed(args.seed)
                with torch.no_grad():
                    new_images = pipe(
                        args.input_text,
                        guidance_scale=7.5,
                        num_inference_steps=50,
                        generator=generator,
                        num_images_per_prompt=args.num_images_per_prompt,
                    ).images
                    if os.path.isfile(os.path.join(tmp_int8_images, "int8.png")):
                        os.remove(os.path.join(tmp_int8_images, "int8.png"))
                    grid = image_grid(new_images, rows=_rows, cols=args.num_images_per_prompt // _rows)
                    grid.save(os.path.join(tmp_int8_images, "int8.png"))
                    fid = fid_score.calculate_fid_given_paths((args.base_images, tmp_int8_images), 1, "cpu", 2048, 8)
                    return fid

            from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
            from neural_compressor.quantization import fit
            accuracy_criterion = AccuracyCriterion(
                higher_is_better=False,
                criterion="relative" if args.is_relative else "absolute",
                tolerable_loss=args.perf_tol)
            quantization_config = PostTrainingQuantConfig(
                approach=args.quantization_approach,
                accuracy_criterion=accuracy_criterion
            )
            model = fit(
                model=pipe.unet,
                conf=quantization_config,
                eval_func=eval_func,
                calib_func=calibration_func,
            )
            setattr(pipe, name, model)
            model.save(args.output_dir)
            logger.info(f"Optimized model {name} saved to: {args.output_dir}.")

    if args.benchmark or args.accuracy_only:

        def b_func(pipe):
            benchmark(pipe, generator)

        if args.int8:
            print("====int8 inference====")
            from neural_compressor.utils.pytorch import load
            checkpoint = os.path.join(args.output_dir)
            pipe.unet = load(checkpoint, model=getattr(pipe, "unet"))
            pipe.unet.eval()
        elif args.bf16:
            if args.ipex:
                print("====ipex bf16 inference====")
                sample = torch.randn(2, 4, 64, 64)
                timestep = torch.rand(1)*999
                encoder_hidden_status = torch.randn(2, 77, 768)
                input_example = (sample, timestep, encoder_hidden_status)
                pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)
                pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)
                pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)
            else:
                print("====bf16 inference====")
                pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
                pipe.unet = pipe.unet.to(torch.bfloat16)
                pipe.vae = pipe.vae.to(torch.bfloat16)
        else:
            print("====fp32 inference====")

        generator = torch.Generator("cpu").manual_seed(args.seed)
        if args.benchmark:
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig

            b_conf = BenchmarkConfig(cores_per_instance=4, num_of_instance=1)
            fit(pipe, config=b_conf, b_func=b_func)
        if args.accuracy_only:
            accuracy(pipe, generator, _rows, args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
