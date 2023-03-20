#MODEL_NAME=resnet12
NAME_EXPERIMENT=miniimagenet_LEAKY_ME
DATASET=miniimagenet
PARENT_DIR=${DATASET}_LEAKY_ME


mkdir -p $PARENT_DIR
for i in  miniimagenet_resnet9s_32x32,resnet9 #miniimagenet_resnet12s_84x84,resnet12
do
    IFS=',' read WEIGHT_NAME MODEL_NAME <<< "${i}"
    echo "processing ${WEIGHT_NAME}.pt";
    for IMAGE_SIZE in 32 64 84 100 
    do 
        DIR_SAVE=${PARENT_DIR}/${IMAGE_SIZE};
        mkdir -p $DIR_SAVE;
        SAVENAME=${DIR_SAVE}/${NAME_EXPERIMENT}_${MODEL_NAME}_${WEIGHT_NAME}_${IMAGE_SIZE}.txt;
        BACKBONEPATH=/homes/a19lavra/Documents/brain-train/weights/${WEIGHT_NAME}.pt;
        python3 main.py --test-dataset ${DATASET}_test --backbone ${MODEL_NAME}  \
        --epochs 1 --training-dataset miniimagenet_train\
        --load-backbone $BACKBONEPATH --test-image-size $IMAGE_SIZE --dataset-path /users/local/datasets/ \
        --few-shot --few-shot-ways 5 --few-shot-queries 15 --few-shot-runs 10000 --few-shot-classifier ncm \
        --feature-maps 32 --few-shot-shots 1  \
        --leaky --feature-processing ME --freeze-backbone --use-strides  2>&1 | tee $SAVENAME;
    done
done


