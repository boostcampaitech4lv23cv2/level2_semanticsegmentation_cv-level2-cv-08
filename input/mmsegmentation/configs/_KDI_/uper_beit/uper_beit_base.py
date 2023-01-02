_base_ = [
    './models/upernet_beit.py', 
    './datasets/dataset.py',
    './schedules/schedule.py', 
    './runtime.py'
]

model = dict(
    pretrained='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth')
