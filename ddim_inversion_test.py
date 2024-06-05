from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    print(pil_img.size)
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
    # model_id = 'stabilityai/stable-diffusion-2-1'
    model_id = "runwayml/stable-diffusion-v1-5"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)
    vae = pipe.vae.eval()
    unet = pipe.unet.eval()

    input_img = load_image(imgname, target_size=(1024, 768)).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, vae)

    inv_steps = 50
    # inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
    #                       # width=input_img.shape[-1], height=input_img.shape[-2],
    #                       output_type='latent', return_dict=False,
    #                       num_inference_steps=inv_steps, latents=latents)

    if verify:
        pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
        noise = torch.randn_like(latents)
        inv_latents = latents + noise

        prompt = "all green"
        image = pipe(prompt=prompt, negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.title(f'with random noise, prompt={prompt}')
        plt.show()
    return inv_latents


if __name__ == '__main__':
    ddim_inversion('./ddim_test.png', num_steps=50, verify=True)
