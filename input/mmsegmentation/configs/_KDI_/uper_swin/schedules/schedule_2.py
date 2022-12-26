optimizer = dict(
    type='AdamW',
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

optimizer_config = dict()

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    min_lr_ratio=1e-6,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=25000)
evaluation = dict(interval=500, metric='mIoU', save_best='mIoU', pre_eval=True, by_epoch=False)
