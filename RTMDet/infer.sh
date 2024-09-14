#python infer.py \
#    --config work_dirs/rtmdet_x_syncbn_fast_8xb32-300e_coco/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
#    --ckpt work_dirs/rtmdet_x_syncbn_fast_8xb32-300e_coco/best_coco_bbox_mAP_epoch_292.pth \
#    --img_dir /data2/wangfj/data/road_damage/test_images \
#    --out_dir /data2/wangfj/data/road_damage/rtmdet_best

python infer.py \
    --config work_dirs/rtmdet_pseudo/rtmdet_x_syncbn_fast_8xb32-300e_coco.py \
    --ckpt work_dirs/rtmdet_pseudo/best_coco_bbox_mAP_epoch_292.pth \
    --img_dir /data2/wangfj/data/road_damage/test_images \
    --out_dir /data2/wangfj/data/road_damage/rtmdet_pseudo_best #\
    # --prefix 
