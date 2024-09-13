import os
import argparse
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config, ConfigDict
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
register_all_modules()
from codetr import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='work_dirs/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco.py',type=str, help="config path")
    parser.add_argument('--ckpt', default='work_dirs/dino-5scale_swin-l_8xb2-12e_coco/epoch_12.pth',type=str, help="ckpt path")
    parser.add_argument('--img_dir', default='/data0/wangfj/road_damage/test_images',type=str, help="image directory")
    parser.add_argument('--save_dir', default='/data0/wangfj/road_damage/annotations',type=str, help="save directory")
    parser.add_argument('--out_dir', default='/data0/wangfj/road_damage/detr_out',type=str, help="output directory")
    parser.add_argument('--device', default='cuda:0',type=str, help="GPU device or CPU")
    parser.add_argument('--thresh', default=.5,type=float, help="confidence threshold")
    parser.add_argument('--prefix', default=None,type=str, help="Prefix")
    parser.add_argument('--tta', action='store_true', help="Whether to use test time augmentation")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = args.config
    checkpoint = args.ckpt
    img_dir = args.img_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    device = args.device
    threshold = args.thresh
    cfg = Config.fromfile(cfg)
    if args.tta:
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
                                      " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
                                         "in config. Can't use tta !"
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline
    # print(cfg)
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    checkpoint_meta = checkpoint.get('meta', {})
    model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
    model.dataset_meta['palette'] = 'random'
    model.cfg = cfg  # save the config in the model for convenience
    model.to(device)
    model.eval()
    #model = init_detector(cfg, checkpoint, device)
    nations = ['Overall', 'India', 'Japan', 'Norway', 'United_States']
    fps = []

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

        for i, nation in enumerate(nations):
            if nation in name or nation == 'Overall':
                fps[i].write(txt)

    for fp in fps:
        fp.close()


if __name__ == "__main__":
    main()

