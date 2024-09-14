#python infer.py \
#	--config work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
#	--ckpt work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/best_coco_bbox_mAP_epoch_16.pth \
#	--img_dir /data0/wangfj/road_damage/test_images \
#	--out_dir /data0/wangfj/road_damage/codetr_best_tta \
#	--tta

python infer.py \
        --config configs/co_detr/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
        --ckpt work_dirs/co-detr_pseudo/best_coco_bbox_mAP_epoch_13.pth \
        --img_dir /data0/wangfj/road_damage/test_images \
        --out_dir /data0/wangfj/road_damage/codetr_pseudo_best

