_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model = dict(
#     type='MaskRCNN',  # The name of detector
#     pretrained=
#     'torchvision://resnet50',  # The ImageNet pretrained backbone to be loaded
#     backbone=dict(  # The config of backbone
#         type='ResNet',  # The type of the backbone, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 for more details.
#         depth=50,  # The depth of backbone, usually it is 50 or 101 for ResNet and ResNext backbones.
#         num_stages=4,  # Number of stages of the backbone.
#         out_indices=(0, 1, 2, 3),  # The index of output feature maps produced in each stages
#         frozen_stages=1,  # The weights in the first 1 stage are fronzen
#         norm_cfg=dict(  # The config of normalization layers.
#             type='BN',  # Type of norm layer, usually it is BN or GN
#             requires_grad=True),  # Whether to train the gamma and beta in BN
#         norm_eval=True,  # Whether to freeze the statistics in BN
#         style='pytorch'),  # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 'caffe' means stride 2 layers are in 1x1 convs.
#     neck=dict(
#         type='FPN',  # The neck of detector is FPN. We also support 'NASFPN', 'PAFPN', etc. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py#L10 for more details.
#         in_channels=[256, 512, 1024, 2048],  # The input channels, this is consistent with the output channels of backbone
#         out_channels=256,  # The output channels of each level of the pyramid feature map
#         num_outs=5),  # The number of output scales
#     rpn_head=dict(
#         type='RPNHead',  # The type of RPN head is 'RPNHead', we also support 'GARPNHead', etc. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12 for more details.
#         in_channels=256,  # The input channels of each input feature map, this is consistent with the output channels of neck
#         feat_channels=256,  # Feature channels of convolutional layers in the head.
#         anchor_generator=dict(  # The config of anchor generator
#             type='AnchorGenerator',  # Most of methods use AnchorGenerator, SSD Detectors uses `SSDAnchorGenerator`. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10 for more details
#             scales=[8],  # Basic scale of the anchor, the area of the anchor in one position of a feature map will be scale * base_sizes
#             ratios=[0.5, 1.0, 2.0],  # The ratio between height and width.
#             strides=[4, 8, 16, 32, 64]),  # The strides of the anchor generator. This is consistent with the FPN feature strides. The strides will be taken as base_sizes if base_sizes is not set.
#         bbox_coder=dict(  # Config of box coder to encode and decode the boxes during training and testing
#             type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9 for more details.
#             target_means=[0.0, 0.0, 0.0, 0.0],  # The target means used to encode and decode boxes
#             target_stds=[1.0, 1.0, 1.0, 1.0]),  # The standard variance used to encode and decode boxes
#         loss_cls=dict(  # Config of loss function for the classification branch
#             type='CrossEntropyLoss',  # Type of loss for classification branch, we also support FocalLoss etc.
#             use_sigmoid=True,  # RPN usually perform two-class classification, so it usually uses sigmoid function.
#             loss_weight=1.0),  # Loss weight of the classification branch.
#         loss_bbox=dict(  # Config of loss function for the regression branch.
#             type='L1Loss',  # Type of loss, we also support many IoU Losses and smooth L1-loss, etc. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56 for implementation.
#             loss_weight=1.0)),  # Loss weight of the regression branch.
#     roi_head=dict(  # RoIHead encapsulates the second stage of two-stage/cascade detectors.
#         type='StandardRoIHead',  # Type of the RoI head. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10 for implementation.
#         bbox_roi_extractor=dict(  # RoI feature extractor for bbox regression.
#             type='SingleRoIExtractor',  # Type of the RoI feature extractor, most of methods uses SingleRoIExtractor. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10 for details.
#             roi_layer=dict(  # Config of RoI Layer
#                 type='RoIAlign',  # Type of RoI Layer, DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79 for details.
#                 out_size=7,  # The output size of feature maps.
#                 sample_num=0),  # Sampling ratio when extracting the RoI features. 0 means adaptive ratio.
#             out_channels=256,  # output channels of the extracted feature.
#             featmap_strides=[4, 8, 16, 32]),  # Strides of multi-scale feature maps. It should be consistent to the architecture of the backbone.
#         bbox_head=dict(  # Config of box head in the RoIHead.
#             type='Shared2FCBBoxHead',  # Type of the bbox head, Refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177 for implementation details.
#             in_channels=256,  # Input channels for bbox head. This is consistent with the out_channels in roi_extractor
#             fc_out_channels=1024,  # Output feature channels of FC layers.
#             roi_feat_size=7,  # Size of RoI features
#             num_classes=80,  # Number of classes for classification
#             bbox_coder=dict(  # Box coder used in the second stage.
#                 type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of methods.
#                 target_means=[0.0, 0.0, 0.0, 0.0],  # Means used to encode and decode box
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),  # Standard variance for encoding and decoding. It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
#             reg_class_agnostic=False,  # Whether the regression is class agnostic.
#             loss_cls=dict(  # Config of loss function for the classification branch
#                 type='CrossEntropyLoss',  # Type of loss for classification branch, we also support FocalLoss etc.
#                 use_sigmoid=False,  # Whether to use sigmoid.
#                 loss_weight=1.0),  # Loss weight of the classification branch.
#             loss_bbox=dict(  # Config of loss function for the regression branch.
#                 type='L1Loss',  # Type of loss, we also support many IoU Losses and smooth L1-loss, etc.
#                 loss_weight=1.0)),  # Loss weight of the regression branch.
#         mask_roi_extractor=dict(  # RoI feature extractor for bbox regression.
#             type='SingleRoIExtractor',  # Type of the RoI feature extractor, most of methods uses SingleRoIExtractor.
#             roi_layer=dict(  # Config of RoI Layer that extracts features for instance segmentation
#                 type='RoIAlign',  # Type of RoI Layer, DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack are also supported
#                 out_size=14,  # The output size of feature maps.
#                 sample_num=0),  # Sampling ratio when extracting the RoI features.
#             out_channels=256,  # Output channels of the extracted feature.
#             featmap_strides=[4, 8, 16, 32]),  # Strides of multi-scale feature maps.
#         mask_head=dict(  # Mask prediction head
#             type='FCNMaskHead',  # Type of mask head, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L21 for implementation details.
#             num_convs=4,  # Number of convolutional layers in mask head.
#             in_channels=256,  # Input channels, should be consistent with the output channels of mask roi extractor.
#             conv_out_channels=256,  # Output channels of the convolutional layer.
#             num_classes=80,  # Number of class to be segmented.
#             loss_mask=dict(  # Config of loss function for the mask branch.
#                 type='CrossEntropyLoss',  # Type of loss used for segmentation
#                 use_mask=True,  # Whether to only train the mask in the correct class.
#                 loss_weight=1.0))))  # Loss weight of mask branch.
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'content/mmdetection/data/coco'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(_delete_=True)
