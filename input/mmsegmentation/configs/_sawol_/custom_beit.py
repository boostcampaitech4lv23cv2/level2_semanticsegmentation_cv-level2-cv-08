_base_ = [
    "../_base_/models/upernet_beit.py",
    "../_base_/datasets/trash_dataset.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_320k.py",
]

model = dict(
    pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
    # pretrained="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth",
    backbone=dict(
        type="BEiT",
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23],
    ),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=11, channels=1024
    ),
    auxiliary_head=dict(in_channels=1024, num_classes=11),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(426, 426)),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95),
)

lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    min_lr_ratio=1e-6,
    periods=[4000, 6000, 10000],
    restart_weights=[1.0, 0.5, 0.3],
    by_epoch=False)

data = dict(samples_per_gpu=3)
optimizer_config = dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=2)

fp16 = dict()