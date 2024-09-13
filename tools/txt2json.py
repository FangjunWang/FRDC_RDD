import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./data',type=str, help="root path of images and labels, include ./JPEGImages and ./labels and classes.txt")
parser.add_argument('--img_dir', default='JPEGImages',type=str, help="path of images")
parser.add_argument('--txt_dir', default='labels',type=str, help="path of txt files")
parser.add_argument('--env', default=None,type=str, help="path of txt files")
parser.add_argument('--env_file', default=None,type=str, help="path of txt files")
parser.add_argument('--save_path', type=str,default='./train.json', help="if not split the dataset, give a path to a json file")
parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
parser.add_argument('--split_by_file', action='store_true', help="define how to split the dataset, include ./train.txt ./val.txt ./test.txt ")

arg = parser.parse_args()

def train_test_val_split_random(img_paths,ratio_train=0.8,ratio_test=0.0,ratio_val=0.2):
    
    assert int(ratio_train+ratio_test+ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    if ratio_test == 0:
        val_img = middle_img
        test_img = []
    else:
        ratio=ratio_val/(1-ratio_train)
        val_img, test_img  =train_test_split(middle_img,test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img

def train_test_val_split_by_files(img_paths, root_dir):
    # 
    phases = ['train', 'val', 'test']
    img_split = []
    for p in phases:
        define_path = os.path.join(root_dir, f'{p}.txt')
        print(f'Read {p} dataset definition from {define_path}')
        assert os.path.exists(define_path)
        with open(define_path, 'r') as f:
            img_paths = f.readlines()
            # img_paths = [os.path.split(img_path.strip())[1] for img_path in img_paths]  #
            img_split.append(img_paths)
    return img_split[0], img_split[1], img_split[2]

    
def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId
    

def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ",root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, arg.txt_dir)                                        
    originImagesDir = os.path.join(root_path, arg.img_dir)
    #with open(os.path.join(root_path, 'classes.txt')) as f:
    #    classes = f.read().strip().split()
    classes = ["__reserved__", "D00", "D10", "D20", "D40", "Repair"]
    indexes = os.listdir(originImagesDir)

    if arg.random_split or arg.split_by_file:
        # 
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        # 
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            
        if arg.random_split:
            print("spliting mode: random split")
            train_img, val_img, test_img = train_test_val_split_random(indexes,0.8,0.0,0.2)
        elif arg.split_by_file:
            print("spliting mode: split by files")
            train_img, val_img, test_img = train_test_val_split_by_files(indexes, root_path)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 0):
            if i == 0:
                continue
            # dataset['categories'].append({'id': i, 'name': cls, 'supercategory': cls, "color": colors[i], "isthing": is_thing[i]})
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': cls})
    
    # 
    if arg.env is not None:
        env = json.load(open(arg.env_file, 'r'))
        print(env)
    ann_id_cnt = 0
    pbar = tqdm(indexes)
    for k, index in enumerate(pbar):
        #
        txtFile = index.replace('.jpg','.txt').replace('.png','.txt')
        if arg.env is not None:
            key = index[:9]
            pbar.set_description(key + ' ' + env[key])
            if env[key] != arg.env:
                continue
        #
        im = cv2.imread(os.path.join(originImagesDir, index))
        height, width, _ = im.shape
        if arg.random_split or arg.split_by_file:
            #
                if index in train_img:
                    dataset = train_dataset
                elif index in val_img:
                    dataset = val_dataset
                elif index in test_img:
                    dataset = test_dataset
        #
        img_id = k
        # img_id = get_image_Id(index)
        dataset['images'].append({'file_name': index,
                                    'id': img_id,
                                    'width': width,
                                    'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            print(f"can not find txt file for {index}")
            continue
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                '''
                x1 = float(label[1])
                y1 = float(label[2])
                x2 = float(label[3])
                y2 = float(label[4])
                
                '''
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                #'''
                #
                cls_id = int(label[0])  + 1  
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                if width == 0 or height == 0:
                    continue
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': img_id,
                    'iscrowd': 0,
                    # mask
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 
    # folder = os.path.join(root_path, 'annotations')
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    if arg.random_split or arg.split_by_file:
        for phase in ['train','val','test']:
            json_name = os.path.join(root_path, '{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, '{}'.format(arg.save_path))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":

    yolo2coco(arg)
