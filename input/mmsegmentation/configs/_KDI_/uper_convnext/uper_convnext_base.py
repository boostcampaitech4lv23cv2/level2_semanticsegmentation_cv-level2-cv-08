_base_ = [
    './models/upernet_convnext.py', 
    './datasets/dataset.py',
    './schedules/schedule.py', 
    './runtime.py'
]
crop_size = (512, 512)
model = dict(
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=11),
    auxiliary_head=dict(in_channels=512, num_classes=11),
)
