from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


@torch.cuda.amp.autocast(enabled=False)
def forward_unet(
    latents,
    t,
    encoder_hidden_states,
    unet,
    weights_dtype,
):
    input_dtype = latents.dtype
    return unet(
        latents.to(weights_dtype),
        t.to(weights_dtype),
        encoder_hidden_states=encoder_hidden_states.to(weights_dtype),
    ).sample.to(input_dtype)


@torch.cuda.amp.autocast(enabled=False)
def encode_cond_images(
    imgs, vae, weights_dtype
):
    input_dtype = imgs.dtype
    imgs = imgs * 2.0 - 1.0
    posterior = vae.encode(imgs.to(weights_dtype)).latent_dist
    latents = posterior.mode()
    uncond_image_latents = torch.zeros_like(latents)
    latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
    return latents.to(input_dtype)


@torch.no_grad()
def ddim_inversion(imgname: str, num_steps: int = 50, verify: Optional[bool] = False) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1',
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae.eval()
    unet = pipe.unet.eval()

    input_img = load_image(imgname).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    weights_dtype = torch.float16
    guidance_scale = 7.5
    condition_scale = 1.5

    image_cond_latents = encode_cond_images(input_img, vae, weights_dtype)

    # inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
    #                       # width=input_img.shape[-1], height=input_img.shape[-2],
    #                       output_type='latent', return_dict=False,
    #                       num_inference_steps=num_steps, latents=latents)

    with torch.no_grad():
        for i, t in enumerate(inverse_scheduler.timesteps):
            # pred noise
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = torch.cat(
                [latent_model_input, image_cond_latents], dim=1
            )

            print(latent_model_input.shape)

            noise_pred = unet(
                latent_model_input.to(weights_dtype),
                t.to(weights_dtype),
                encoder_hidden_states=None,
            ).sample.to(dtype)

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                3
            )
            noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + condition_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample

    # verify
    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder='scheduler')
        image = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        # ax[2].imshow(latents[0][0].cpu().numpy())
        plt.show()
    return latents


if __name__ == '__main__':
    ddim_inversion('./poike.png', num_steps=250, verify=True)
