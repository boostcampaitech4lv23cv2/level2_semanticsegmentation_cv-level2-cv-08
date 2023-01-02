_base_ = [
    './models/segformer_mit-b0.py',
    './datasets/dataset.py',
    './schedules/schedule.py',
    './runtime.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(pretrained=checkpoint, decode_head=dict(num_classes=11))

