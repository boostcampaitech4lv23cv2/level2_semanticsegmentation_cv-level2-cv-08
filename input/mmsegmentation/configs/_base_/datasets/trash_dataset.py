# dataset settings
dataset_type = "CustomDataset"
data_root = "/opt/ml/input/data"

classes = [
    "Background",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]
palette = [
    [0, 0, 0],
    [192, 0, 128],
    [0, 128, 192],
    [0, 128, 64],
    [128, 0, 0],
    [64, 0, 128],
    [64, 0, 192],
    [192, 128, 64],
    [192, 192, 128],
    [64, 64, 128],
    [128, 0, 192],
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
img_scale = (512, 512)
crop_size = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    # dict(
    #     type="Albu",
    #     transforms=[
    #         # dict(type='ToSepia'),
    #         dict(type="ToGray"),
    #         # dict(type='RandomGamma',
    #         #     gamma_limit=[20, 180],
    #         #     p=0.5),
    #         # dict(type='RandomBrightnessContrast',
    #         #     brightness_limit=[-0.3, 0.3],
    #         #     contrast_limit=[-0.3, 0.3],
    #         #     p=0.2),
    #         # dict(type='CLAHE',
    #         #     clip_limit=40.0,
    #         #     ),
    #     ],
    #     keymap=dict(img="image", gt_semantic_seg="mask"),
    #     update_pad_shape=False,
    # ),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
valid_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        palette=palette,
        classes=classes,
        img_dir=data_root + "/mmseg/img_dir/train",
        ann_dir=data_root + "/mmseg/ann_dir/train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        palette=palette,
        classes=classes,
        img_dir=data_root + "/mmseg/img_dir/val",
        ann_dir=data_root + "/mmseg/ann_dir/val",
        pipeline=valid_pipeline,
    ),
    test=dict(
        type=dataset_type,
        palette=palette,
        classes=classes,
        img_dir=data_root + "/mmseg/img_dir/test",
        pipeline=test_pipeline,
    ),
)