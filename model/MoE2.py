
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

#Changing the above to accomodate noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        print(threshold_if_in.shape, noisy_values.shape)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        # load = self._prob_in_top_k(logits,noise_logits,noise,top_k_logits).sum(0)
        return router_output, indices


#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, loss_coeff=0.01):
        super(MoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.loss_coeff = loss_coeff

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    

    def forward(self, x):
        gating_output, indices = self.router(x)

        importance = gating_output.sum(0)

        expert_ops = torch.stack([expert(x) for expert in self.experts],dim=-1)

        return (expert_ops * gating_output.unsqueeze(2)).sum(dim=-1), self.loss_coeff*(self.cv_squared(importance))#+ self.cv_squared(load))
        # final_output = torch.zeros_like(x)

        # # Reshape inputs for batch processing
        # flat_x = x.view(-1, x.size(-1))
        # flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # # Process each expert in parallel
        # for i, expert in enumerate(self.experts):
        #     # Create a mask for the inputs where the current expert is in top-k
        #     expert_mask = (indices == i).any(dim=-1)
        #     flat_mask = expert_mask.view(-1)

        #     if flat_mask.any():
        #         expert_input = flat_x[flat_mask]
        #         expert_output = expert(expert_input)

        #         # Extract and apply gating scores
        #         gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
        #         weighted_output = expert_output * gating_scores

        #         # Update final output additively by indexing and adding
        #         final_output[expert_mask] += weighted_output.squeeze(1)

        # return final_output
