import torch.nn as nn

from lib.monoDTR.visualDet3D.networks.detectors.monodtr_core import MonoDTRCore
from lib.monoDTR.visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from lib.monoDTR.visualDet3D.networks.heads.depth_losses import bin_depths, DepthFocalLoss

class MonoDTR(nn.Module):
    def __init__(self, network_cfg,dataset_cfg):
        super(MonoDTR, self).__init__()

        # Core
        self.mono_core = MonoDTRCore()

        # Head
        self.bbox_head = AnchorBasedDetection3DHead(network_cfg['head'],dataset_cfg)
        self.depth_loss = DepthFocalLoss(96)


#    def train_forward(self, left_images, annotations, P2, depth_gt=None):
#        
#        features, depth = self.mono_core(dict(image=left_images, P2=P2))
#        
#        depth_output   = depth
#
#        cls_preds, reg_preds = self.bbox_head(
#                dict(
#                    features=features,
#                    P2=P2,
#                    image=left_images
#                )
#            )
#
#        anchors = self.bbox_head.get_anchor(left_images, P2)
#
#        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)
#        
#        depth_gt = bin_depths(depth_gt, mode = "LID", depth_min=1, depth_max=80, num_bins=96, target=True)
#
#        if reg_loss.mean() > 0 and not depth_gt is None and not depth_output is None:
#            
#            depth_gt = depth_gt.unsqueeze(1)
#            depth_loss = 1.0 * self.depth_loss(depth_output, depth_gt)
#            loss_dict['depth_loss'] = depth_loss
#            reg_loss += depth_loss
#
#            self.depth_output = depth_output.detach()
#        else:
#            loss_dict['depth_loss'] = torch.zeros_like(reg_loss)
#        return cls_loss, reg_loss, loss_dict

    def test_forward(self, left_images, P2):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        features, _ = self.mono_core(dict(image=left_images, P2=P2))
        
        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=features,
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        
        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) >= 3:
            raise NotImplementedError
        else:
            return self.test_forward(*inputs)