_base_ = [
    './models/upernet_convnext.py', 
    './datasets/dataset.py',
    './schedules/schedule.py', 
    './runtime.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # noqa

crop_size = (512, 512)
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=11,
    ),
    auxiliary_head=dict(in_channels=768, num_classes=11),
)
