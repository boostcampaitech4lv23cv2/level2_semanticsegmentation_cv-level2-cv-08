# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=1e-4, by_epoch=True)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=651*50)
checkpoint_config = dict(by_epoch=False, interval=651*5)
evaluation = dict(interval=651, metric='mIoU', pre_eval=True)

# runtime custom settings
# runner = dict(type="EpochBasedRunner", max_epochs=200)
# checkpoint_config = dict(interval=5)
# evaluation = dict(interval=1, metric="mIoU", pre_eval=True)