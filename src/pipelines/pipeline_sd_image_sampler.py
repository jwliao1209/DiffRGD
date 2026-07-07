from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torchvision.transforms.functional import to_tensor
from diffusers import StableDiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg


class StableDiffusionImageSamplerPipeline(StableDiffusionPipeline):
    """Stable Diffusion pipeline with inference-time guidance in latent space.

    Supports DPS, MPGD, DSG, and DiffRGD. At every denoising step, the predicted
    clean latent is decoded by the VAE, the guidance loss is computed on the
    decoded image against `reference_image`, and its gradient steers the latent.
    """

    @torch.no_grad()
    def prepare_image(self, image, device, dtype):
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image).unsqueeze(0)
        return image.to(device=device, dtype=dtype)  # shape: (1, 3, H, W), range: [0, 1]

    def _projection(self, grad, direction):
        """Project grad onto the tangent space of the sphere at the current point."""
        normal_grad = (grad * direction).sum() * direction / (torch.linalg.norm(direction) ** 2)
        return grad - normal_grad

    def _retraction(self, sample, direction, center, radius):
        """Map the updated point back onto the sphere of the given center and radius."""
        direction = sample - center + direction
        return center + radius * direction / torch.linalg.norm(direction)

    def _predict_noise(self, latents, t, prompt_embeds, timestep_cond, added_cond_kwargs):
        """UNet forward pass with classifier-free guidance.

        Returns (noise_pred, noise_pred_uncond, noise_pred_text); the last two are
        None when classifier-free guidance is disabled.
        """
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if not self.do_classifier_free_guidance:
            return noise_pred, None, None

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0.0:
            # Based on Section 3.4 of https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
        return noise_pred, noise_pred_uncond, noise_pred_text

    def _decode_original_image(self, original_latents, generator):
        return self.vae.decode(
            original_latents / self.vae.config.scaling_factor,
            return_dict=True,
            generator=generator,
        ).sample

    def _prev_timestep(self, t):
        return t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

    def _ddim_mean_and_std(self, t, original_latents, noise_pred, eta):
        """DDIM posterior mean mu_t and standard deviation sigma_t at timestep t."""
        t_prev = self._prev_timestep(t)
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
        variance = self.scheduler._get_variance(t, t_prev)
        std_dev_t = eta * variance ** 0.5
        latents_mean = alpha_prod_t_prev ** 0.5 * original_latents + \
            (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * noise_pred
        return latents_mean, std_dev_t

    @torch.no_grad()
    def __call__(
        self,
        method: str = "diffrgd",
        reference_image: Optional[torch.Tensor] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        inner_max_iter: int = 1,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        loss_fn: Optional[nn.Module] = None,
        guidance_interval: int = 1,
        step_size: float = 0.1,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """Generate an image guided by `loss_fn(decoded_x0, reference_image)`.

        Arguments follow `StableDiffusionPipeline.__call__`, with the additional
        guidance arguments:

        Args:
            method: Guidance method, one of "dps", "mpgd", "dsg", "diffrgd".
            reference_image: Conditioning image passed to the guidance loss.
            loss_fn: Guidance loss computed on the decoded predicted clean image.
            step_size: Guidance strength eta_t.
            guidance_interval: Apply guidance every N denoising steps.
            inner_max_iter: Number of inner RGD iterations (diffrgd only).
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier-free guidance, concatenate the unconditional and text
        # embeddings into a single batch to avoid two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        reference_image = self.prepare_image(
            image=reference_image,
            device=device,
            dtype=self.unet.dtype,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.config.time_cond_proj_dim,
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            with torch.enable_grad():
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    if method == "dps":
                        latents = latents.clone().detach().requires_grad_(True)
                        noise_pred, _, _ = self._predict_noise(
                            latents, t, prompt_embeds, timestep_cond, added_cond_kwargs
                        )
                        outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                        prev_latents = outputs.prev_sample

                        if t.item() > 10:
                            original_image = self._decode_original_image(outputs.pred_original_sample, generator)
                            loss = loss_fn(original_image.clamp(-1.0, 1.0), reference_image)
                            grad = torch.autograd.grad(outputs=loss, inputs=latents)[0]
                            prev_latents = prev_latents - step_size * grad
                        latents = prev_latents.detach()

                    elif method == "mpgd":
                        noise_pred, _, _ = self._predict_noise(
                            latents, t, prompt_embeds, timestep_cond, added_cond_kwargs
                        )
                        outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                        original_latents = outputs.pred_original_sample.requires_grad_(True)
                        original_image = self._decode_original_image(original_latents, generator)

                        if t.item() > 1:
                            loss = loss_fn(original_image.clamp(-1.0, 1.0), reference_image)
                            grad = torch.autograd.grad(outputs=loss, inputs=original_latents)[0]
                            original_latents = original_latents - step_size * grad

                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        t_prev = self._prev_timestep(t)
                        alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
                        epsilon = (latents - (alpha_prod_t ** 0.5) * original_latents) / (1 - alpha_prod_t) ** 0.5
                        prev_latents = (alpha_prod_t_prev ** 0.5) * original_latents + ((1 - alpha_prod_t_prev) ** 0.5) * epsilon
                        latents = prev_latents.detach()

                    elif method == "dsg":
                        latents = latents.clone().detach().requires_grad_(True)
                        noise_pred, _, _ = self._predict_noise(
                            latents, t, prompt_embeds, timestep_cond, added_cond_kwargs
                        )
                        outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                        prev_latents = outputs.prev_sample
                        original_latents = outputs.pred_original_sample

                        if i % guidance_interval == 0 and t.item() > 1:
                            latents_mean, std_dev_t = self._ddim_mean_and_std(t, original_latents, noise_pred, eta)
                            original_image = self._decode_original_image(original_latents, generator)

                            loss = loss_fn(original_image, reference_image)
                            grad = torch.autograd.grad(outputs=loss, inputs=latents)[0]
                            _, c, h, w = latents.shape
                            r = torch.sqrt(torch.tensor(c * h * w)) * std_dev_t
                            eps = 1e-8
                            d_star = -r * grad / (torch.linalg.norm(grad) + eps)
                            d_sample = prev_latents - latents_mean
                            mix_direction = d_sample + step_size * (d_star - d_sample)
                            prev_latents = latents_mean + r * mix_direction / (torch.linalg.norm(mix_direction) + eps)
                        latents = prev_latents.detach()

                    elif method == "diffrgd":
                        noise_pred, noise_pred_uncond, noise_pred_text = self._predict_noise(
                            latents, t, prompt_embeds, timestep_cond, added_cond_kwargs
                        )
                        outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                        prev_latents = outputs.prev_sample
                        original_latents = outputs.pred_original_sample

                        if i % guidance_interval == 0 and t.item() > 1:
                            t_prev = self._prev_timestep(t)
                            latents_mean, std_dev_t = self._ddim_mean_and_std(t, original_latents, noise_pred, eta)
                            radius = torch.linalg.norm(prev_latents - latents_mean)

                            for _ in range(inner_max_iter):
                                prev_latents = prev_latents.clone().detach().requires_grad_(True)
                                noise_pred, noise_pred_uncond, noise_pred_text = self._predict_noise(
                                    prev_latents, t_prev, prompt_embeds, timestep_cond, added_cond_kwargs
                                )
                                original_latents = self.scheduler.step(
                                    noise_pred,
                                    t_prev,
                                    prev_latents,
                                    return_dict=True,
                                    **extra_step_kwargs,
                                ).pred_original_sample
                                original_image = self._decode_original_image(original_latents, generator)

                                loss = loss_fn(original_image, reference_image)
                                grad = torch.autograd.grad(outputs=loss, inputs=prev_latents)[0]
                                riemannian_grad = self._projection(grad, prev_latents - latents_mean)

                                # Scale the step size by the ratio between the CFG
                                # correction magnitude and the Riemannian gradient
                                # magnitude. With an empty prompt the correction is
                                # zero, so fall back to the plain step size.
                                adaptive_step_size = step_size
                                if noise_pred_text is not None:
                                    correction = noise_pred_text - noise_pred_uncond
                                    correction_scale = (correction * correction).mean().sqrt() * guidance_scale
                                    if correction_scale > 0:
                                        adaptive_step_size = (
                                            correction_scale
                                            / (riemannian_grad * riemannian_grad).mean().sqrt() * step_size
                                        )
                                prev_latents = self._retraction(
                                    prev_latents, -adaptive_step_size * riemannian_grad, latents_mean, radius
                                )
                        latents = prev_latents.detach()

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
