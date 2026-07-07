from typing import List, Optional, Tuple, Union

import torch
from diffusers import DDIMPipeline
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms.functional import to_tensor

from src.methods.admmdiff import ADMMDiff
from src.methods.diffrgd import DiffRGD
from src.methods.dps import DPS
from src.methods.dsg import DSG
from src.methods.mpgd import MPGD
from src.pipelines.utils import ImagePipelineOutput


class ImageSamplerPipeline(DDIMPipeline):
    """DDIM pipeline with inference-time guidance (DPS, MPGD, DSG, ADMMDiff, DiffRGD).

    At each denoising step, the selected guidance method refines the latent
    toward the reference image with the given guidance loss.
    """

    @torch.no_grad()
    def prepare_image(self, image, device=None, dtype=None):
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image).unsqueeze(0)
        return image.to(device=device, dtype=dtype)

    def _build_solver(
        self,
        method: str,
        loss_fn,
        step_size: float,
        guidance_interval: int,
        inner_max_iter: int,
        generator,
        verbose: bool,
    ):
        common = dict(
            loss_fn=loss_fn,
            step_size=step_size,
            scheduler=self.scheduler,
            model=self.unet,
            generator=generator,
            verbose=verbose,
        )
        if method == "dps":
            return DPS(guidance_interval=guidance_interval, **common)
        if method == "mpgd":
            return MPGD(guidance_interval=guidance_interval, inner_max_iter=inner_max_iter, **common)
        if method == "dsg":
            return DSG(guidance_interval=guidance_interval, **common)
        if method == "admmdiff":
            return ADMMDiff(inner_max_iter=inner_max_iter, **common)
        if method == "diffrgd":
            return DiffRGD(guidance_interval=guidance_interval, inner_max_iter=inner_max_iter, **common)
        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _to_uint8_frame(x: torch.Tensor):
        x = (((x + 1) / 2) * 255).clamp(0, 255)
        return x.detach().cpu().squeeze().numpy().transpose(1, 2, 0).astype("uint8")

    @torch.no_grad()
    def __call__(
        self,
        method: str = "diffrgd",
        reference_image: Optional[torch.Tensor] = None,
        step_size: float = 0.1,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        inner_max_iter: int = 1,
        rho: float = 1.0,
        loss_fn: Optional[torch.nn.Module] = None,
        guidance_interval: int = 1,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        solver = self._build_solver(
            method, loss_fn, step_size, guidance_interval, inner_max_iter, generator, verbose,
        )

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        reference_image = self.prepare_image(
            reference_image,
            device=self._execution_device,
            dtype=self.unet.dtype,
        )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # ADMMDiff maintains an auxiliary variable z and a dual variable across steps
        z_image = image.clone().detach() if method == "admmdiff" else None
        dual_vec = torch.zeros_like(image) if method == "admmdiff" else None

        self.scheduler.set_timesteps(num_inference_steps)

        self.xt_traj = []
        self.x0t_traj = []

        with torch.enable_grad():
            for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                model_output = self.unet(image, t).sample

                output, original_sample = solver.step(
                    sample=image,
                    z_sample=z_image,
                    dual_vec=dual_vec,
                    model_output=model_output,
                    reference_image=reference_image,
                    t=t,
                    rho=rho,
                    eta=eta,
                    use_clipped_model_output=use_clipped_model_output,
                    index=i,
                )

                if method == "admmdiff":
                    image, z_image, dual_vec = output
                else:
                    image = output

                self.xt_traj.append(self._to_uint8_frame(image))
                self.x0t_traj.append(self._to_uint8_frame(original_sample))

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image, xt_traj=self.xt_traj, x0t_traj=self.x0t_traj)
