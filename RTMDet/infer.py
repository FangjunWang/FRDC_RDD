import os
import argparse
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='work_dirs/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco.py', type=str, help="config path")
    parser.add_argument('--ckpt', default='work_dirs/dino-5scale_swin-l_8xb2-12e_coco/epoch_12.pth', type=str, help="ckpt path")
    parser.add_argument('--img_dir', default='/data0/wangfj/road_damage/test_images', type=str, help="image directory")
    parser.add_argument('--save_dir', default=None, type=str, help="save directory")
    parser.add_argument('--out_dir', default='/data0/wangfj/road_damage/detr_out', type=str, help="output directory")
    parser.add_argument('--device', default='cuda:0', type=str, help="GPU device or CPU")
    parser.add_argument('--thresh', default=.5, type=float, help="confidence threshold")
    parser.add_argument('--prefix', default=None, type=str, help="Prefix for image name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = args.config
    checkpoint = args.ckpt
    img_dir = args.img_dir
    save_dir = args.save_dir
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device = args.device
    threshold = args.thresh
    model = init_detector(cfg, checkpoint, device)
    nations = ['Overall', 'India', 'Japan', 'Norway', 'United_States']
    fps = []
    
    if save_dir is not None:
        for nation in nations:
            txt_path = os.path.join(save_dir, nation + '.txt')
            fp = open(txt_path, 'w')
            fps.append(fp)

    for name in tqdm(os.listdir(img_dir)):
        if args.prefix is not None:
            if args.prefix not in name:
                continue
        txt = name + ','
        img = os.path.join(img_dir, name)
        result = inference_detector(model, img).pred_instances
        out_path = os.path.join(out_dir, name.replace('.jpg', '.txt'))
        out = open(out_path, 'w')
        for item in result:
            bbox, label, score = item["bboxes"], item["labels"][0], item["scores"]
            left, top, right, bottom = bbox[0]
            out.write(f'{int(label) + 1} {int(left)} {int(top)} {int(right)} {int(bottom)} {float(score)}\n')
            if score < threshold:
                continue
            txt += f'{int(label) + 1} {int(left)} {int(top)} {int(right)} {int(bottom)} '
        txt += '\n'
        out.close()
        if save_dir is not None:
            for i, nation in enumerate(nations):
                if nation in name or nation == 'Overall':
                    fps[i].write(txt)


if __name__ == "__main__":
    main()

