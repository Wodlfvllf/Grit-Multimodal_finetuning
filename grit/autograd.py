import torch


class KFACAutogradFunction(torch.autograd.Function):
    """Custom autograd function to capture activations and gradients for K-FAC."""

    @staticmethod
    def forward(ctx, module, output, input):
        ctx.module = module
        ctx.save_for_backward(input.detach())
        return output

    @staticmethod
    def backward(ctx, grad_wrt_output):
        module = ctx.module
        input, = ctx.saved_tensors
        manager = module.grit_manager
        
        if not module.training:
            return None, grad_wrt_output, None

        # Per-module gating to avoid starving most modules when many hooks fire per step
        tick = manager.cov_update_tick.get(module, 0) + 1
        manager.cov_update_tick[module] = tick
        if tick % manager.config.grit_cov_update_freq != 0:
            return None, grad_wrt_output, None

        with torch.no_grad():
            # 1) activations -> rank space via A^T
            a = input.reshape(-1, input.shape[-1])
            if a.shape[0] > 0 and 'default' in module.lora_A:
                lora_A_T = module.lora_A['default'].weight.data.T.to(device=a.device, dtype=a.dtype, non_blocking=True)
                projected_a = a @ lora_A_T
                a_cov_sample = projected_a.T @ projected_a  # on-device
                
                # Ensure current_cov is on the same device as a_cov_sample
                current_cov = manager.a_covs[module].to(a_cov_sample.device)
                
                n = manager.num_samples_a[module]
                new_n = n + projected_a.shape[0]
                if new_n > 0:
                    updated_cov = (current_cov.float() * n + a_cov_sample.float()) / new_n
                    manager.a_covs[module].copy_(updated_cov.to(dtype=manager.a_covs[module].dtype))
                    manager.num_samples_a[module] = new_n

            # 2) output grads -> rank space via B
            g = grad_wrt_output.reshape(-1, grad_wrt_output.shape[-1])
            if g.shape[0] > 0 and 'default' in module.lora_B:
                lora_B = module.lora_B['default'].weight.data.to(device=g.device, dtype=g.dtype, non_blocking=True)
                projected_g = g @ lora_B
                g_cov_sample = projected_g.T @ projected_g  # on-device
                
                # Ensure current_cov is on the same device as g_cov_sample
                current_cov = manager.g_covs[module].to(g_cov_sample.device)
                
                n = manager.num_samples_g[module]
                new_n = n + projected_g.shape[0]
                if new_n > 0:
                    updated_cov = (current_cov.float() * n + g_cov_sample.float()) / new_n
                    manager.g_covs[module].copy_(updated_cov.to(dtype=manager.g_covs[module].dtype))
                    manager.num_samples_g[module] = new_n

        return None, grad_wrt_output, None
