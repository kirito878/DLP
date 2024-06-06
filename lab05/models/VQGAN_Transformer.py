import torch
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F

# TODO2 step1: design the MaskGIT model


class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])

        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(
            configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True)
        model = model.eval()
        return model

# TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # zq : b,c,h,w
        # codebook_indices : b*c
        zq, codebook_indices, _ = self.vqgan.encode(x)

        # print("zq",zq.shape)
        # print("codebook_indices",codebook_indices.shape)

        # reshpae codebook_indices : b,c
        codebook_indices = codebook_indices.view(zq.shape[0], -1)
        # print("codebook_indices",codebook_indices.shape)
        return zq, codebook_indices
        # raise Exception('TODO2 step1-1!')

# TODO2 step1-2:
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.

        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        print(f'mode: {mode}')
        if mode == "linear":
            return lambda r: 1 - r
            raise Exception('TODO2 step1-2!')
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
            raise Exception('TODO2 step1-2!')
        elif mode == "square":
            return lambda r: 1 - r ** 2
            raise Exception('TODO2 step1-2!')
        elif mode == "sqrt":
            return lambda r: 1 - np.sqrt(r)
        else:
            raise NotImplementedError

# TODO2 step1-3:
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)
        
        r = math.floor(self.gamma(np.random.uniform())*z_indices.shape[1])

        _, sample = torch.rand(
            z_indices.shape, device=z_indices.device).topk(r, dim=1)

        # mask : b,c
        mask = torch.zeros(z_indices.shape, dtype=torch.bool,
                           device=z_indices.device)

        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * \
            torch.ones_like(z_indices, device=z_indices.device)

        # a_indices : b,c
        a_indices = mask * masked_indices + (~mask) * z_indices

        # logits : b,c,1025
        logits = self.transformer(a_indices)

        # raise Exception('TODO2 step1-3!')
        return logits, z_indices

# TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, mask_num, ratio):
        # raise Exception('TODO3 step1-1!')
        # Set masked tokens to a special index (1024)
        z_indices[mask_bc] = 1024

        logits = self.transformer(z_indices)
        # Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = F.softmax(logits, dim=-1)

        # FIND MAX probability for each token value
        max_probs, max_indices  = torch.max(logits, dim=-1)


        # predicted probabilities add temperature annealing gumbel noise as confidence
        # gumbel noise
        g = -torch.log(-torch.log(torch.rand_like(max_probs)))
        temperature = self.choice_temperature * (1 - ratio)
        confidence = max_probs  + temperature * g

        # hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        # sort the confidence for the rank
        # define how much the iteration remain predicted tokens by mask scheduling
        # At the end of the decoding process, add back the original token values that were not masked to the predicted tokens

        # Set the probability of non-masked tokens to infinity
        max_probs[~mask_bc] = float('inf')
        _, sorted_indices = torch.sort(confidence)
        max_indices[~mask_bc] = z_indices[~mask_bc]

        mask_count = math.floor(ratio * mask_num)
        mask_bc[:, sorted_indices[:, mask_count:]] = False

        return max_indices, mask_bc


__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
