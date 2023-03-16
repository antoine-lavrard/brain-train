
DATASET_NAME=miniimagenet
BACKBONE_NAME=resnet9
RESOLUTION=32
DEVICE=cuda:0 # the 

python main.py --dataset ${DATASET_NAME} --steps "[['manifold mixup'],['rotations']]" --backbone ${BACKBONE_NAME} \
--feature-processing ME --batch-size 376 --epochs 1500 --cosine --lr 0.1 --gamma 0.9 --few-shot \
--save-backbone ${DATASET_NAME}_${BACKBONE_NAME}_${RESOLUTION}x${RESOLUTION}.pt --training-image-size ${RESOLUTION} --test-image-size ${RESOLUTION} \
--dataset-path /users/local/datasets/ --save-features-prefix ${DATASET_NAME}_${BACKBONE_NAME}_${RESOLUTION}x${RESOLUTION}.pt \
--device ${DEVICE} --skip-epochs 1450 --deterministic --leaky --feature-maps 32 --use-strides


