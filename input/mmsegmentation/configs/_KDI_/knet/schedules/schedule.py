# optimizer
# optimizer = dict(
#     constructor='LearningRateDecayOptimizerConstructor',
#     type='AdamW',
#     lr=5e-5,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg={
#         'decay_rate': 0.9,
#         'decay_type': 'stage_wise',
#         'num_layers': 12
#     })
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

# # fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # fp16 placeholder
# fp16 = dict()

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    min_lr_ratio=1e-6,
    periods=[4000, 6000, 10000],
    restart_weights=[1.0, 0.5, 0.3],
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=25000)
evaluation = dict(interval=500, metric='mIoU', save_best='mIoU', pre_eval=True, by_epoch=False)
