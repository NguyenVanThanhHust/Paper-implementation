from detectron2.config import CfgNode as CN

def add_tridentnet_config(cfg):
    """
    build config for trident net
    """
    _C = cfg
    
    _C.MODEL.TRIDENT = CN()

    # number of branches for Trident Net
    _C.MODEL.NUM_BRANCH = 3


I   # dilation for each branch
    _C.MODEL.TRIDENT.BRANCH_DILATIONS = [1, 2, 3]
    
    # stage for applying trident blocks
    _C.MODEL.TRINDET.TRIDENT_STAGE = "res4"
    
    # test branch index for fast inference
    _C.MODEL.TRIDENT.TEST_BRANCH_IDX = 1
