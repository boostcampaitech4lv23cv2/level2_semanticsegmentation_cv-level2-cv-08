import os
import json
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import albumentations as A

from torch.utils.data import Dataset
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
        self.transform = A.Compose([
                    A.RandomScale(scale_limit=0.3, p=1.0),
                    A.RandomRotate90(0.5),
                    A.PadIfNeeded(512, 512, border_mode=0),
                    A.Resize(512, 512),
                ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))

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
        masks = []
        bboxes = []
        
        # 위치 제한 객체 없을 시
        if self.cp_loc: 
            if [1 for bbox in ori['bboxes'] if bbox[4] in self.cp_loc] == []:
                return ori
        
        # TODO: p 통해서 확률 조정
        random_class = np.random.choice(self.cp_obj)
        random_id = np.random.choice(self.categories[random_class])
        random_ann = self.coco.loadAnns(int(random_id))[0]
        img_id = random_ann['image_id']

        image = self.get_image(img_id)
        select_mask = self.coco.annToMask(random_ann)
        
        for ann in self.coco.getAnnIds(img_id):
            load_ann = self.coco.loadAnns(ann)[0]
            mask = self.coco.annToMask(load_ann)
            for i,j in zip([1,1,-1,-1],[1,0,1,0]):
                temp = mask + np.roll(select_mask, shift=i, axis=j)
                if any(map(any, temp > 1)):
                    masks.append(mask)
                    bboxes.append(load_ann['bbox']+[load_ann['category_id']])
                    select_mask = temp > 0
                    break
        
        item = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        output = self.transform(**item)
        temp = []
        for bboxes in output['bboxes']:
            temp.append([bboxes[0],bboxes[1],bboxes[0]+bboxes[2],bboxes[1]+bboxes[3],bboxes[4]])
        o_x, o_y,*_ = map(int,np.min(temp, axis=0))
        _,_,o_w, o_h,*_ = map(int,np.max(temp, axis=0))
        o_w, o_h = o_w-o_x, o_h-o_y
        
        new_mask = np.zeros((512,512)).astype(np.uint8)
        for i, m in enumerate(output['masks']):
            new_mask[m > 0] = output['bboxes'][i][-1]
        
        # 위치 선정
        if self.cp_loc: # 특정 객체 위에만 paste
            loc_bboxes = [bbox for bbox in ori['bboxes'] if bbox[4] in self.cp_loc]
            r_x, r_y, *_ = loc_bboxes[np.random.choice(len(loc_bboxes))]
            r_x = np.random.randint(min(512-o_w, r_x), 513-o_w)
            r_y = np.random.randint(min(512-o_h, r_y), 513-o_h)
        elif len(self.cp_loc) == 0: # 랜덤 위치
            r_x = np.random.randint(0,513-o_w)
            r_y = np.random.randint(0,513-o_h)
        else:
            raise Exception("잘못된 위치 입력")
        
        # 이미지 붙여넣기
        ori_img = ori['image']
        cut_img = output['image'][o_y:o_y+o_h, o_x:o_x+o_w]
        cut_mask = new_mask[o_y:o_y+o_h, o_x:o_x+o_w]
        mask_range = cut_mask > 0

        ori_img[r_y:r_y+o_h, r_x:r_x+o_w][mask_range] = cut_img[mask_range]
        
        ori['new_masks'][r_y:r_y+o_h, r_x:r_x+o_w][mask_range] = cut_mask[mask_range]
        ori['masks'].extend(output['masks'])
        ori['image'] = ori_img
        ori['bboxes'].extend(output['bboxes'])
                
        return ori
    
if __name__ == "__main__":
    dataset = CocoDataset('../data', 'train.json')
    if not os.path.exists('cp_data/img'):
        os.makedirs('cp_data/img')
    if not os.path.exists('cp_data/ann'):
        os.makedirs('cp_data/ann')
        
    for i in tqdm(range(len(dataset))):
        data = dataset[i].copy()
        for cnt in range(1):
            data = dataset.get_random_obj(data)
        cv2.imwrite(f'cp_data/img/{str(i).zfill(4)}.jpg', cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'cp_data/ann/{str(i).zfill(4)}.png', data['new_masks'].astype(np.uint8))