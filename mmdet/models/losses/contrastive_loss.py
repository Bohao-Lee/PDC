"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.reweight_func = reweight_func

    def forward(self, features, labels):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        features = features.squeeze(1)
        assert features.shape[0] == labels.shape[0]
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()
        # print("test", features.shape, features.T.shape)
        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)
        

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        loss = -per_label_log_prob

        # coef = self._get_reweight_func(self.reweight_func)(ious)

        # loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

