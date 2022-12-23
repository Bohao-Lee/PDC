import torch
import torch.nn as nn
import torch.nn.functional as F
from optparse import Values
import torch
import numpy as np

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.models.losses.contrastive_loss import SupConLoss
import fvcore.nn.weight_init as weight_init

class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


@HEADS.register_module()
class PDCRoIHead(StandardRoIHead):
    """RoIHead with multi-scale feature modulator on the input of bbox head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 mae_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 dataset='coco',
                 split=1,
                 backbone='vit_b',
                 ):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dataset = dataset
        self.split = split
        self.backbone = backbone

        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()
        
        if mae_head is not None:
            self.mae_head = build_head(mae_head)
            self.with_mae_head = True
        else:
            self.with_mae_head = False

        self.std_x, self.std_y, self.std_w, self.std_h = self.get_std(dataset=self.dataset, split=self.split, backbone=self.backbone)
        
        self.noising_num = 50 #50 20 10 
        self.noising_scalar = 0.1 # noising of gt 
        self.noising_loss_scalar = 0.1
        self.contrastive_loss_fun = SupConLoss(temperature=0.2)

        self.contrastive_scalar = 0.1
        # for vit-b
        self.contrastive_head = ContrastiveHead(dim_in=512, feat_dim=256)


    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        if isinstance(bbox_roi_extractor, list):
            self.ms_bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor[0])
            self.ss_bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor[1])
            self.mfm_factor = nn.Parameter(torch.zeros(bbox_head['in_channels'], requires_grad=True), requires_grad=True)
            self.mfm_fc = nn.Conv2d(in_channels=256,out_channels=bbox_head['in_channels'],kernel_size=1)
            self.with_mfm = True
        else:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
            self.with_mfm = False
        self.bbox_head = build_head(bbox_head)



    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            if self.with_mfm:
                self.ms_bbox_roi_extractor.init_weights()
                self.ss_bbox_roi_extractor.init_weights()
            else:
                self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights(pretrained)
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def generate_noising_proposal(self, gt_bboxes):
        noising_gt_bboxes = []
        for current_image_gt_bboxes in gt_bboxes:
            current_image_noising_gt_bboxes = []
            if current_image_gt_bboxes.shape[0] == 0:
                current_image_gt_bboxes = (torch.rand(1,4) * 100).to(current_image_gt_bboxes.device)
            for gt_bbox in current_image_gt_bboxes:
                bbox_point = torch.tensor([(gt_bbox[0] + gt_bbox[2])/2, (gt_bbox[1]+gt_bbox[3])/2]).unsqueeze(0).repeat(self.noising_num,1).to(gt_bbox.device)
                bbox_h_w = torch.tensor([gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]]).unsqueeze(0).repeat(self.noising_num,1).to(gt_bbox.device)
                
                # Gaussian distribution:
                x_noise = torch.normal(0, self.std_x, size=(self.noising_num,1)).to(gt_bbox.device)
                y_noise = torch.normal(0, self.std_y, size=(self.noising_num,1)).to(gt_bbox.device)
                w_noise = torch.normal(0, self.std_w, size=(self.noising_num,1)).to(gt_bbox.device)
                h_noise = torch.normal(0, self.std_h, size=(self.noising_num,1)).to(gt_bbox.device)

                point_noise = torch.cat([x_noise, y_noise], dim=1)
                h_w_noise = torch.cat([w_noise, h_noise], dim=1)
                noising_bbox_point = bbox_h_w * point_noise + bbox_point
                noising_bbox_h_w = bbox_h_w * h_w_noise + bbox_h_w
                current_noising_gt_bbox = torch.stack([
                                            noising_bbox_point[:,0]-noising_bbox_h_w[:,0]/2,
                                            noising_bbox_point[:,1]-noising_bbox_h_w[:,1]/2,
                                            noising_bbox_point[:,0]+noising_bbox_h_w[:,0]/2,
                                            noising_bbox_point[:,1]+noising_bbox_h_w[:,1]/2]
                                            ,dim=1)
                current_image_noising_gt_bboxes.append(current_noising_gt_bbox)
            current_image_noising_gt_bboxes = torch.cat(current_image_noising_gt_bboxes, dim=0)
            noising_gt_bboxes.append(current_image_noising_gt_bboxes)
        return noising_gt_bboxes

    
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      vit_feat=None,
                      img=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            
            noising_proposals_list = self.generate_noising_proposal(gt_bboxes)
            sampling_results = []
            noising_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                noising_assign_result = self.bbox_assigner.assign(
                    noising_proposals_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                noising_sampling_result = self.bbox_sampler.sample(
                    noising_assign_result,
                    noising_proposals_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                noising_sampling_results.append(noising_sampling_result)

        losses = dict()

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._noising_bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
                                                   
            noising_bbox_results = self._noising_bbox_forward_train(x, noising_sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            
            denoising_bbox = {}
            for key, values in noising_bbox_results['loss_bbox'].items():
                if 'loss' in key:
                    denoising_bbox.update({"noising_" + key: values * self.noising_loss_scalar})
                else:
                    denoising_bbox.update({"noising_" + key: values})

            losses.update(bbox_results['loss_bbox'])
            losses.update(denoising_bbox)

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            
        # mae head forward and loss
        if self.with_mae_head:
            loss_rec = self.mae_head(vit_feat, img)
            losses.update(loss_rec)

        # print(losses)
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
    
    def _noising_bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        # bbox_feats: tensor [B,C,7,7]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, x = self.bbox_head.forward_noising(bbox_feats)
        # x: tensor [B,256]
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, x

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _noising_bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        bbox_results, roi_feature = self._noising_bbox_forward(x, rois)
        proposal_mask = torch.sum(bbox_targets[3], dim=1) > 0
        
        ground_truth_rois = bbox2roi([bbox_targets[2]])
        _, ground_truth_roi_feature = self._noising_bbox_forward(x, ground_truth_rois)

        proposal_bbox = torch.cat([res.bboxes for res in sampling_results], dim=0)
        target_bbox = bbox_targets[2]
        # print("test", proposal_bbox.shape, bbox_targets[2].shape, ious.shape, ious)
        if torch.sum(proposal_mask.to(torch.int)) > 0:
            feature_mask = proposal_mask.reshape(-1,1).repeat(1, roi_feature.shape[1])
            roi_feature = torch.masked_select(roi_feature, feature_mask).reshape(-1, feature_mask.shape[1])
            ground_truth_roi_feature = torch.masked_select(ground_truth_roi_feature, feature_mask).reshape(-1, feature_mask.shape[1])
            contrastive_label = torch.masked_select(bbox_targets[0], proposal_mask)
            class_num = list(set(bbox_targets[0].cpu().detach().numpy().tolist()))
            roi_feature = self.contrastive_head(roi_feature)
            contrastive_loss = self.contrastive_loss_fun(roi_feature.unsqueeze(1), contrastive_label)
            # print("contrastive_loss", contrastive_loss)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                                bbox_results['bbox_pred'], rois,
                                                *bbox_targets)
            
            loss_bbox['loss_noising_contrastive'] = contrastive_loss * self.contrastive_scalar * int(len(class_num) > 1)
        else:
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                                bbox_results['bbox_pred'], rois,
                                                *bbox_targets)
            roi_feature = self.contrastive_head(roi_feature)
            contrastive_loss = self.contrastive_loss_fun(roi_feature.unsqueeze(1), bbox_targets[0])
            loss_bbox['loss_noising_contrastive'] = contrastive_loss * 0

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results


    def get_std(self, dataset='voc', split=1, backbone='vit_s'):
        if backbone == 'vit_s':
            if dataset == 'voc':
                if split == 1:
                    std_x = 0.036745
                    std_y = 0.024628
                    std_w = 0.051736
                    std_h = 0.038616
                elif split == 2:
                    std_x = 0.034175
                    std_y = 0.024625
                    std_w = 0.048201
                    std_h = 0.037519
                elif split == 3:
                    std_x = 0.06091
                    std_y = 0.041583
                    std_w = 0.085972
                    std_h = 0.062982
                else:
                    raise("Error split! must be 1 or 2 or 3 for voc")
            elif dataset == 'coco':
                std_x = 0.057854
                std_y = 0.060303
                std_w = 0.082206
                std_h = 0.085963
            else:
                raise("Error dataset! must be voc or coco")
        elif backbone == 'vit_b':
            if dataset == 'voc':
                if split == 1:
                    std_x = 0.047011
                    std_y = 0.030668
                    std_w = 0.064949
                    std_h = 0.049197
                elif split == 2:
                    std_x = 0.036111
                    std_y = 0.024807
                    std_w = 0.050466
                    std_h = 0.037804
                elif split == 3:
                    std_x = 0.043781
                    std_y = 0.028429
                    std_w = 0.06102
                    std_h = 0.045307
                else:
                    raise("Error split! must be 1 or 2 or 3 for voc")
            elif dataset == 'coco':
                std_x = 0.050455
                std_y = 0.05086
                std_w = 0.071394
                std_h = 0.073416
            else:
                raise("Error dataset! must be voc or coco")
        else:
            raise("Error backbone! must be vit_s or vit_b")
            
        return std_x, std_y, std_w, std_h


    