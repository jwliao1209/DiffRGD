"""
DiffRGD: Inference-Time Diffusion Guidance Through Riemannian Gradient Descent.

At each DDIM sampling step, the latent x_{t-1} ~ N(mu_t, sigma_t^2 I) is polar-
decomposed as x_{t-1} = mu_t + sigma_t * r * u (Proposition 1 in the paper). Fixing
the sampled radius sigma_t * r defines a spherical manifold S_{t,r}, and the
guidance loss is minimized on this manifold with K Riemannian Gradient Descent
steps (Algorithm 1 in the paper).
"""

import torch
from torch import nn


class DiffRGD(nn.Module):
    def __init__(
        self,
        loss_fn,
        step_size,
        scheduler,
        model,
        inner_max_iter=3,
        generator=None,
        guidance_interval=1,
        verbose=False,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.step_size = step_size
        self.scheduler = scheduler
        self.model = model
        self.generator = generator
        self.inner_max_iter = inner_max_iter
        self.guidance_interval = guidance_interval
        self.verbose = verbose

    def _projection(self, grad, direction):
        """Project grad onto the tangent space of the sphere at the current point."""
        normal_grad = (grad * direction).sum() * direction / (torch.linalg.norm(direction) ** 2)
        return grad - normal_grad

    def _retraction(self, sample, direction, center, radius):
        """Map the updated point back onto the sphere of the given center and radius."""
        direction = sample - center + direction
        return center + radius * direction / torch.linalg.norm(direction)

    def step(self, sample, model_output, reference_image, t, eta, index, use_clipped_model_output=True, *args, **kwargs):
        outputs = self.scheduler.step(
            model_output,
            t,
            sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=self.generator,
        )
        prev_sample = outputs.prev_sample
        t_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        if index % self.guidance_interval == 0 and t_prev > 0:
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
            variance = self.scheduler._get_variance(t, t_prev)
            std_dev_t = eta * variance ** 0.5

            # DDIM mean mu_t; the sphere center of S_{t,r}
            sample_mean = alpha_prod_t_prev ** 0.5 * outputs.pred_original_sample + \
                (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output

            # Sampled radius sigma_t * r; fixed during the inner RGD iterations
            radius = torch.linalg.norm(prev_sample - sample_mean)

            for k in range(self.inner_max_iter):
                prev_sample = prev_sample.clone().detach().requires_grad_(True)
                original_sample = self.scheduler.step(
                    self.model(prev_sample, t_prev).sample,
                    t_prev,
                    prev_sample,
                    eta=eta,
                    use_clipped_model_output=use_clipped_model_output,
                    generator=self.generator,
                ).pred_original_sample

                loss = self.loss_fn(original_sample, reference_image)
                grad = torch.autograd.grad(outputs=loss, inputs=prev_sample)[0]
                riemannian_grad = self._projection(grad, prev_sample - sample_mean)
                prev_sample = self._retraction(prev_sample, -self.step_size * riemannian_grad, sample_mean, radius)

                if self.verbose:
                    print(f"iteration: {k + 1}/{self.inner_max_iter}")
                    print(f"radius: {radius.item()}")
                    print(f"step size: {self.step_size}")
                    print(f"loss: {loss.item()}")
                    print(f"gradient norm: {torch.linalg.norm(grad).item()}")

        if t_prev > 0:
            original_sample = self.scheduler.step(
                self.model(prev_sample, t_prev).sample,
                t_prev,
                prev_sample,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=self.generator,
            ).pred_original_sample
        else:
            original_sample = prev_sample

        return prev_sample.detach(), original_sample.detach()
