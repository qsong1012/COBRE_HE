import torch

def zero_padding_first_dim(t, n):
    size = list(t.size())
    size[0] = n
    o = torch.zeros(size, dtype=t.dtype, device=t.device)
    o[:t.size(0),] = t[:n]
    return o