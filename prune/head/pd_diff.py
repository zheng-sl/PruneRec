import torch


def kl_divergence(base_pd, mask_pd):
    epsilon = 1e-12
    base_pd = base_pd + epsilon
    mask_pd = mask_pd + epsilon

    now_kl_divergence = torch.sum(base_pd * torch.log(base_pd / mask_pd))

    return now_kl_divergence

def kl_divergence_per_token(base_pd, mask_pd):
    epsilon = 1e-12
    base_pd = base_pd + epsilon
    mask_pd = mask_pd + epsilon

    now_kl_divergence = torch.sum(base_pd * torch.log(base_pd / mask_pd), dim=-1).mean()

    return now_kl_divergence