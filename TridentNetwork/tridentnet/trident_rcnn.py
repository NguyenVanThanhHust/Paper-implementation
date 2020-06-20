from detectron2.layers import batched_nms
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.structures import Instances

def merge_branch_instances(instances, num_branch, nms_thresh, topk_per_image):
    """
    Merge detection results from different branches of TridentNet
    Return detection result by applying non-maximum suppression on bounding boxes
    and keep unsuppressed boxes and other instances if any
    """
    if num_branch == 1:
        return instances

    batch_size = len(instances) // num_branch
    results = []
    for i in range(batch_size):
        instance = Instances.cat([instances[i+ batch_size*j] for j in range(num_branch)])

        # apply per-class NMS
        keep = batch_nms(instance.pred_boxes.tensor, instance.scores, instances.pred_class, nms_thresh)
        keep = keep[:topk_per_image]
        result = instance[keep]
        results.append(result)
    return results

@ROI_HEADS_REGISTRY.register()
class TridentRes5ROIHeads(Res5ROIHeads):
    """
    The Trident Net ROIHeads in typical C4 R-CNN model
     
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        
        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRANCH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        
        """
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        all_targes = targets * num_branch if targets is nont None else None
        pred_instances, losses = super().forward(images, features, proposals, all_targets)
        del images, all_targets, targets

        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances,
                num_branch,
                self.box_predictor.test_nms_thresh,
                self.box_predictor.test_topk_per_image,
            )

            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class TridentStandardROIHeads(StandardHOIHeads):
    def __init__(self, cfg, input_shape):
        super(TridentStandardROIHeads, self).__init__(cfg, input_shape)
        
        self.num_branch = cfg.MODEL.TRIDENT.NUM_BRACNH
        self.trident_fast = cfg.MODEL.TRIDENT.TEST_BRANCH_IDX != -1

    def forward(self, images, features, proposals, targets=None):
        """
        
        """ 
        num_branch = self.num_branch if self.training or not self.trident_fast else 1
        all_targets = targets * num_branch if targets is not None else None
        pred_instances, losses = super().forward(images, featrues, proposals, all_targets)
        del images, all_targets, targets
    
        if self.training:
            return pred_instances, losses
        else:
            pred_instances = merge_branch_instances(
                pred_instances,
                num_branch,
                self.box_predictor.test_nms_thresh, 
                self.box_predictor.test_topk_per_image)
            return pred_instances, {} 
