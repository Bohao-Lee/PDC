# base training

# bash tools/dist_train.sh configs/imted/few_shot/imted_ss_faster_rcnn_vit_base_2x_base_training_coco.py 8 \
#     --cfg-options \
#     data.train.data_root=../../MSCOCO2014 \
#     data.val.data_root=../../MSCOCO2014 \
#     data.test.data_root=../../MSCOCO2014 \
#     --work-dir ../cvpr/few_shot/imted_ss_faster_rcnn_vit_base_2x_base_training_coco \
 
# sleep 2m

# pip install -v -e .

# 10shot

# bash tools/dist_train.sh configs/imted/few_shot_pdc/imted_ss_faster_rcnn_vit_base_2x_finetuning_10shot_coco.py 8 \
#     --cfg-options \
#     data.train.data_root=../MSCOCO2014 \
#     data.val.data_root=../MSCOCO2014 \
#     data.test.data_root=../MSCOCO2014 \
#     --work-dir ../cvpr/few_shot/imted_ss_faster_rcnn_vit_base_2x_finetuning_10shot_coco \


# sleep 2m

# 30shot
bash tools/dist_train.sh configs/imted/few_shot_pdc/imted_ss_faster_rcnn_vit_base_2x_finetuning_30shot_coco.py 8 \
    --cfg-options \
    data.train.data_root=../MSCOCO2014 \
    data.val.data_root=../MSCOCO2014 \
    data.test.data_root=../MSCOCO2014 \
    --work-dir ../cvpr/few_shot/imted_ss_faster_rcnn_vit_base_2x_finetuning_30shot_coco \

