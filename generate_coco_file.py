import os
import json
import shutil
from PIL import Image
import matplotlib
import matplotlib.pylab as plt
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.mask import encode, decode, area, toBbox


def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def ans_prttier():
    with open('./answer.json') as fp:
        d = json.load(fp)
        res = json.dumps(d, indent=4, sort_keys=False)
        print(res)

    with open('./answer_prettyprint.json', 'w') as fp:
        fp.write(res)

def split_train_val():
    src = './train/'
    dst = './val/'

    if not os.path.exists(os.path.join('./', 'val')):
        os.makedirs(os.path.join('./', 'val'))

    train_filenames = next(os.walk(src))[1]
    for img_id, file_name in tqdm(enumerate(train_filenames), total=len(train_filenames)):
        # Split 3 data from train to val
        if img_id < 3:
            print(f'Move {src + file_name} to {dst}')
            shutil.move(src + file_name, dst)

def get_mask_info(mask_img):
    """
    Description:
        Get RLE & (area of RLE) & (left, top, width, height) from grayscale mask.
    args: 
        mask_img: grayscale image of cv2 (pixel value: 0-255)
    return: 
        (encoded, (x, y, w, h))
        encoded: RLE of mask_img
        (x, y, w, h): left, top, width and height
    """

    # Get RLE of segmentation mask
    encoded = encode(np.asfortranarray(mask_img))
    rle_area = area(encoded)
    [x, y, w, h] = toBbox(encoded)
    bbox = [x, y, w, h]
    print(type(x))
    # Convert bytes to str
    trans = encoded['counts'].decode("utf-8")
    encoded['counts'] = trans
    
    return encoded, rle_area, bbox

def combine_coco(images_coco, categories_coco, annotations_coco):
    result = {
        "images": images_coco,
        "categories": categories_coco,
        "annotations": annotations_coco,
    }
    return result
    """
    COCO annotation format
    {
        "images": [image],
        "annotations": [annotation],
        "categories": [category]
    }


    image = {
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
    }

    annotation = {
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }

    categories = [{
        "id": int,
        "name": str,
        "supercategory": str,
    }]

    """

    """
    EX. 
    {
        "images": [
            {
                "height": 1000,
                "width": 1000,
                "id": 0,
                "file_name": "TCGA-18-5592-01Z-00-DX1.png"
            }
        ],
        "categories": [
            {
                "supercategory": "nucleus",
                "id": 0,
                "name": "nucleus"
            }
        ],
        "annotations": [
            {
                "id": 1
                "image_id": 0,
                "category_id": 0,
                "segmentation": {
                    'size': [1000, 1000], 
                    'counts': b'Q[]17mn06M3L3M3M3M3N2M3M2O2N100O100N20000O100O1000O100000001N10001N101N101N1O2N2N3K4K6G\\Vkk0'
                },
                "area": 1558.0,
                "bbox": [101.0, 118.0, 136.0, 167.0],
                "iscrowd": 0,
            }
        ]
    }
    """



def get_image_coco_fmt(id, width, height, file_name):
    image = {
        "id": id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }
    return image

def get_category_coco_fmt():
    categories = {
        "id": 1,
        "name": "nucleus",
        "supercategory": "nucleus",
    }
    return categories

def get_annotation_coco_fmt(mask_id, image_id, RLE, area, bbox):
    annotation = {
        "id": mask_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": RLE,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }
    return annotation

def produce_trainval_coco_json(train_path = './train/', val=False):
    # produce annotated json file in coco format (for instance segmentation)
    train_filenames = next(os.walk(train_path))[1]

    images_coco, categories_coco, annotations_coco = [], [], []
    mask_counts = 0

    category = get_category_coco_fmt()
    categories_coco.append(category)

    for img_id, file_name in tqdm(enumerate(train_filenames), total=len(train_filenames)):
        pic_path = train_path + file_name + '/images/' + file_name + '.png'
        mask_path = train_path + file_name + '/masks/'
        img = cv2.imread(pic_path)[...,::-1]
        img_h, img_w, _ = img.shape

        image = get_image_coco_fmt(img_id, img_w, img_h, file_name + '.png')
        images_coco.append(image)

        for mask_filename in next(os.walk(mask_path))[2]:
            # print(mask_filename)
            mask = cv2.imread(mask_path + mask_filename, cv2.IMREAD_GRAYSCALE)
            encoded, rle_area, bbox = get_mask_info(mask)

            annotation = get_annotation_coco_fmt(mask_counts, img_id, encoded, rle_area, bbox)
            annotations_coco.append(annotation)
            # print(annotation)
            mask_counts += 1

    result = combine_coco(images_coco, categories_coco, annotations_coco)

    json_filename = './val_coco.json' if val else './train_coco.json'
    result_coco = json.dumps(result, indent=4, sort_keys=False, default=convert)
    with open(json_filename, 'w') as fp:
        fp.write(result_coco)

def produce_test_coco_json(img_order_ref='./test_img_ids.json'):
    images_coco, categories_coco, annotations_coco = [], [], []
    category = get_category_coco_fmt()
    categories_coco.append(category)

    with open(img_order_ref) as fp:
        d = json.load(fp)
        res = json.dumps(d, indent=4, sort_keys=False)
        images_coco = json.loads(res)

    result = combine_coco(images_coco, categories_coco, annotations_coco)
    result_coco = json.dumps(result, indent=4, sort_keys=False, default=convert)
    with open('./test_coco.json', 'w') as fp:
        fp.write(result_coco)

             


# def produce_test_coco_json(path='./test/'):
#     # (Didn't notice there has already been provided. )
#     test_filenames = os.listdir(path)

#     images_coco, categories_coco, annotations_coco = [], [], []

#     category = get_category_coco_fmt()
#     categories_coco.append(category)
    
#     for img_id, file_name in enumerate(test_filenames):
#         pic_path = path + file_name
#         # mask_path = train_path + file_name + '/masks/'
#         img = cv2.imread(pic_path)[...,::-1]
#         img_h, img_w, _ = img.shape

#         image = get_image_coco_fmt(img_id, img_w, img_h, file_name)
#         images_coco.append(image)
    
#     result = combine_coco(images_coco, categories_coco, annotations_coco)
#     result_coco = json.dumps(result, indent=4, sort_keys=False, default=convert)
#     with open('./test_coco.json', 'w') as fp:
#         fp.write(result_coco)

def organize_pic_folder(path, val=False):
    dst_folder = './val_pic/' if val else './train_pic/'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    filenames = next(os.walk(path))[1]
    for img_id, file_name in enumerate(filenames):
        pic_path = path + file_name + '/images/' + file_name + '.png'
        shutil.copy(pic_path, dst_folder)
        print(f'{pic_path} 2 {dst_folder}')

def main():
    train_path = './train/'
    val_path = './val/'
    test_path = './test/'

    #produce_trainval_coco_json(val_path, val=True)
    produce_trainval_coco_json(train_path)
    # produce_test_coco_json(test_path)

    organize_pic_folder(train_path)
    organize_pic_folder(val_path, val=True)

if __name__ == '__main__':
    # split_train_val()
    main()
    # produce_test_coco_json()
