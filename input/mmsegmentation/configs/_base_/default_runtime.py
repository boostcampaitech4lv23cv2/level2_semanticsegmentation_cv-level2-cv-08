import wandb

# yapf:disable
log_config = dict(
    interval=99,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="MMSegWandbHook",
            init_kwargs=dict(
                entity="8bit_seg", project="model-test", name="beit"
            ),
            # log_checkpoint=True,
            # log_checkpoint_metadata=True,
            num_eval_images=100,
        )
    ],
)

# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True