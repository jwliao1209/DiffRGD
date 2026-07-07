'''
Decoupling Training-Free Guided Diffusion by ADMM
Paper: https://arxiv.org/abs/2411.12773
GitHub: https://github.com/youyuan-zhang/ADMMDiff
'''

import torch
from torch import nn


class ADMMDiff(nn.Module):
    def __init__(
        self,
        loss_fn,
        scheduler,
        model,
        inner_max_iter=10,
        step_size=2.4,
        generator=None,
        verbose=False,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.step_size = step_size
        self.scheduler = scheduler
        self.model = model
        self.inner_max_iter = inner_max_iter
        self.generator = generator
        self.verbose = verbose

    def step(self, z_sample, dual_vec, reference_image, t, eta, rho=1, use_clipped_model_output=True, *args, **kwargs):
        # x-subproblem
        x_sample = z_sample - dual_vec / rho
        x_prev_sample = self.scheduler.step(
            self.model(x_sample, t).sample,
            t,
            x_sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=self.generator,
        ).prev_sample

        # z-subproblem
        z_sample = z_sample.clone().detach().requires_grad_(True)
        for _ in range(self.inner_max_iter):
            z_pred_original_sample = self.scheduler.step(
                self.model(z_sample, t).sample,
                t,
                z_sample,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=self.generator,
            ).pred_original_sample
            loss = self.loss_fn(z_pred_original_sample, reference_image)
            z_grad = torch.autograd.grad(outputs=loss, inputs=z_sample)[0]
            z_sample = z_sample - self.step_size * (z_grad + rho * (z_sample - x_prev_sample - dual_vec))
            z_sample = z_sample.detach().requires_grad_(True)

        z_prev_sample = z_sample

        # u-subproblem
        dual_vec = dual_vec + rho * (x_prev_sample - z_prev_sample)

        if self.verbose:
            print(f"step size: {self.step_size}")
            print(f"loss: {loss.item()}")
            print(f"gradient norm: {torch.linalg.norm(z_grad).item()}")
            print(f"|x-z|: {torch.linalg.norm(x_prev_sample - z_prev_sample).item()}")
            print(f"dual vector norm: {torch.linalg.norm(dual_vec).item()}")

        return (x_prev_sample.detach(), z_prev_sample.detach(), dual_vec.detach()), z_pred_original_sample.detach()
