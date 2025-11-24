import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal

# ht = fϕ(ht−1, zt−1, at−1)
class RecurrentModel(nn.Module):
    def __init__(self, hidden_size, stochastic_size, action_size, input_size):
        super().__init__()

        self.nonlinear   = nn.Tanh()
        # linear used to combine info into a new feature space
        self.linear    = nn.Linear(stochastic_size + action_size, input_size)
        self.recurrent = nn.GRUCell(input_size, hidden_size)

    def forward(self, prev_hidden, prev_stochastic, prev_action):
        x = torch.cat((prev_stochastic, prev_action), dim=-1)
        x = self.nonlinear(self.linear(x))
        h = self.recurrent(x, prev_hidden) 
        return h # shape (batch_size, hidden_size)
    
# copied from "learning architecture/encoder.ipynb" 
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, model_dim: int = 1024):
        super().__init__()

        chs = max(16, model_dim // 16)

        # want to increase the feature channel (dim=1) as we compress the image (dim 2 and 3)
        # we begin with (B, feature_chs, H, W)
        # want feature_chs to increase to incorporate more semantic meaning
        # while reducing the visual map by compression so that semantic wholeness is acquired
        # i.e., if we compress the image enough, the weights have to learn more 'bigger picture' features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,   chs,   kernel_size=4, stride=2, padding=1), nn.SiLU(), 
            nn.Conv2d(chs,           chs*2, kernel_size=4, stride=2, padding=1), nn.SiLU(),     
            nn.Conv2d(chs*2,         chs*4, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(chs*4,         chs*8, kernel_size=4, stride=2, padding=1), nn.SiLU()
        )
        # we expect the flattened output, e.g., all dimensions except the batch multiplied together
        # assumes that the end output will have H, W = 4, 4
        self.proj = nn.Linear(8 * chs * 4 * 4, model_dim)

    def forward(self, x): # expects a 64 x 64 img
        h = self.conv(x)
        # just in case, enforce 4,4 
        if h.shape[-1] != 4 or h.shape[-2] != 4:
            h = F.adaptive_avg_pool2d(h, (4,4))
        # we need to flatten before putting into linear
        h = h.flatten(1) # flatten from dim=1 to dim=-1
        logits = self.proj(h)
        return logits # shape (batch_size, model_dim)

# the "Posterior" is a part of the Encoder of the RSSM in the DreamerV3 paper
# it gives the stochastic representation, zt ∼ qϕ(zt | ht, xt)
# where ht = hidden state, xt = encoded observation, and zt = stochastic representation
class Posterior(nn.Module):
    def __init__(self, input_size, hidden_size: int = 200, stochastic_length: int = 16, stochastic_classes: int = 16):
        super().__init__()
        #self.config = config

        self.stochastic_length  = stochastic_length
        self.stochastic_classes = stochastic_classes
        self.stochastic_size    = stochastic_classes * stochastic_length

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.SiLU(),
            nn.Linear(hidden_size, self.stochastic_size)
        )
        
        self.uniform_mix = 0.01 # config.uniform_mix

    def forward(self, x):
        B = x.shape[0]
        L = self.stochastic_length
        C = self.stochastic_classes
        
        # get probs
        raw_logits = self.mlp(x).reshape(B, L, C)
        probs = F.softmax(raw_logits, dim=-1)

        # mix with uniform for regularization to prevent overconfidence
        uniform = torch.ones_like(probs) / C
        probs_unimix = (1 - self.uniform_mix) * probs + self.uniform_mix * uniform

        # get tau
        tau = 1.0 # getattr(self.config, "gumbel_temp", 1.0)

        # from uniform mixed probs to logits
        logits = torch.log(probs_unimix.clamp_min(1e-20)) # clamp to ensure no log(0) -> -inf
        sample = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        return sample.view(-1, self.stochastic_size), logits

        
# at ∼ πθ(at | st), where st = {ht, zt}
class Actor(nn.Module):
    # st_size == hidden_size + stochastic_size
    def __init__(self, action_size, st_size, action_low, action_high, device, hidden_size=400): # hidden_size = mlp # neurons in hidden layer
        super().__init__()
        action_size *= 2 # for splitting into mean and std 

        self.mlp = nn.Sequential(
            nn.Linear(st_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size)
        )

        # for scaling the action to the proper range after sampling
        self.register_buffer("action_scale", ((torch.tensor(action_high, device=device) - torch.tensor(action_low, device=device)) / 2.0))
        self.register_buffer("action_bias", ((torch.tensor(action_high, device=device) + torch.tensor(action_low, device=device)) / 2.0))

    def forward(self, x, training=False):
        # original soft actor-critic implementation:
        # clip min and max of std to prevent unstable gradients and premature convergence
        log_std_min, log_std_max = -5, 2

        # get mean and log std
        mean, log_std = self.mlp(x).chunk(2, dim=-1)

        # smooth squashing + rescaling to confine std
        # tanh + 1 -> [0, 2]
        # log_std -> [-5, 2]
        log_std = log_std_min + (log_std_max - log_std_min)/2 * (torch.tanh(log_std) + 1)
        std     = torch.exp(log_std)

        dist        = Normal(mean, std)  # action gaussian distribution
        sample      = dist.sample()      # stop gradient flow
        sample_tanh = torch.tanh(sample) # bound the distribution from (-inf, inf) -> (-1, 1)

        # scale the action to the env range via affine transformation
        action = sample_tanh * self.action_scale + self.action_bias

        # TODO: build training condition
        if training:
            return action
        else:
            return action

