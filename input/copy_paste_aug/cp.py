import os
import json
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CocoDataset(Dataset):
    def __init__(self, root, filename,
                cp_obj:list=[],
                cp_loc:list=[]):
        self.root = root
        self.coco = COCO(os.path.join(root,filename))
        self.ids = self.coco.getImgIds()
        
        self.cp_obj = cp_obj if cp_obj else self.coco.getCatIds() # 복사해올 객체
        self.cp_loc = cp_loc # 특정 객체 위에만 paste
        self.class_num = len(self.coco.getCatIds())
        
        ids = []
        categories = {i:[] for i in range(1,11)}
        
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
                for ann in anno:
                    categories[ann['category_id']].append(ann['id'])
        self.ids = ids
        self.categories = categories

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        target = sorted(target, key=lambda idx : idx['area'], reverse=True)

        image = self.get_image(img_id)
        
        masks = []
        bboxes = []
        new_masks = np.zeros((512,512))
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])
            pixel_value = obj['category_id']
            new_masks[self.coco.annToMask(obj) == 1] = pixel_value

        new_masks = new_masks.astype(np.int8)
        #pack outputs into a dict
        #bboxes = x,y,w,h,class,idx
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes,
            'new_masks': new_masks
        }
        
        return output
    
    def get_image(self, img_id):
        path = self.coco.loadImgs(img_id)[0]['file_name']   
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_random_obj(self, ori):
        # 위치 제한 객체 없을 시
        if self.cp_loc: 
            if [1 for bbox in ori['bboxes'] if bbox[4] in self.cp_loc] == []:
                return ori
        
        random_class = np.random.choice(self.cp_obj)
        random_id = np.random.choice(self.categories[random_class])
        random_ann = self.coco.loadAnns(int(random_id))[0]
        transform = A.Compose([
                    A.RandomScale(scale_limit=0.3, p=1.0),
                    A.RandomRotate90(0.5),
                    A.PadIfNeeded(512, 512, border_mode=0),
                    A.Resize(512, 512),
                ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))

        image = self.get_image(random_ann['image_id'])
        masks = [self.coco.annToMask(random_ann)]
        bboxes = [random_ann['bbox'] + [random_ann['category_id']]]
        item = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        output = transform(**item)
        o_x, o_y, o_w, o_h = map(int,output['bboxes'][0][:4])
        
        # 위치 선정
        if self.cp_loc: # 특정 객체 위에만 paste
            loc_bboxes = [bbox for bbox in ori['bboxes'] if bbox[4] in self.cp_loc]
            r_x, r_y, *_ = loc_bboxes[np.random.choice(len(loc_bboxes))]
            r_x = np.random.randint(min(511-o_w, r_x), 512-o_w)
            r_y = np.random.randint(min(511-o_h, r_y), 512-o_h)
        elif len(self.cp_loc) == 0: # 랜덤 위치
            r_x = np.random.randint(0,512-o_w)
            r_y = np.random.randint(0,512-o_h)
        else:
            raise Exception("잘못된 위치 입력")
        
        # 이미지 붙여넣기
        ori_img = ori['image']
        cut_img = output['image'][o_y:o_y+o_h, o_x:o_x+o_w]
        cut_mask = output['masks'][0][o_y:o_y+o_h, o_x:o_x+o_w]
        mask_range = cut_mask == 1
        new_mask = np.zeros((512,512)).astype(np.uint8)
        new_mask[r_y:r_y+o_h, r_x:r_x+o_w][mask_range] = 1
        ori_img[r_y:r_y+o_h, r_x:r_x+o_w][mask_range] = cut_img[mask_range]
        
        # for i in range(len(ori['masks'])):
        #     ori['masks'][i][new_mask == 1] = 0
        
        ori['new_masks'][new_mask == 1] = random_class
        ori['masks'].append(new_mask)
        ori['image'] = ori_img
        ori['bboxes'].append([r_x,r_y,o_w,o_h,random_class, ori['bboxes'][-1][-1]+1])
        
        return ori
    
if __name__ == "__main__":
    dataset = CocoDataset('../data', 'train.json', [2,3,4,5,6,7,8,9,10],[8])

    if not os.path.exists('cp_data/img'):
        os.makedirs('cp_data/img')
    if not os.path.exists('cp_data/ann'):
        os.makedirs('cp_data/ann')
        
    for i in tqdm(range(len(dataset))):
        data = dataset[i].copy()
        for cnt in range(2):
            data = dataset.get_random_obj(data)
        cv2.imwrite(f'cp_data/img/{str(i).zfill(4)}.jpg', cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'cp_data/ann/{str(i).zfill(4)}.png', data['new_masks'].astype(np.uint8))