MODEL_NAME=resnet12
NAME_EXPERIMENT=miniimagenet_LEAKY_ME
DATASET=miniimagenet
PARENT_DIR=${DATASET}_LEAKY_ME

# easy-smallmini1,45,pt1 easy-cifar1,64,pt1 easy-smallcifar1,45,pt1 easy-tinycifar1,32,pt1
# miniimagenet_resnet12_trainsize32,32,pt easy-mini1,64,pt1
mkdir -p $PARENT_DIR
for i in easy-tinymini1,32,pt1 easy-smallmini1,45,pt1
do
    IFS=',' read WEIGHT_NAME FEATURE_MAP EXT <<< "${i}"
    echo "processing ${WEIGHT_NAME}.${EXT}";
    for IMAGE_SIZE in 32 64 84 100 
    do 
        DIR_SAVE=${PARENT_DIR}/${IMAGE_SIZE};
        mkdir -p $DIR_SAVE;
        SAVENAME=${DIR_SAVE}/${NAME_EXPERIMENT}_${MODEL_NAME}_${WEIGHT_NAME}_${IMAGE_SIZE}.txt;
        BACKBONEPATH=/homes/a19lavra/Documents/brain-train/weights/${WEIGHT_NAME}.$EXT;
        python3 main.py --test-dataset ${DATASET}_test --backbone resnet12  \
        --epochs 1 --training-dataset miniimagenet_train\
        --load-backbone $BACKBONEPATH --test-image-size $IMAGE_SIZE --dataset-path /users/local/datasets/ \
        --few-shot --few-shot-ways 5 --few-shot-queries 15 --few-shot-runs 10000 --few-shot-classifier ncm \
        --feature-maps $FEATURE_MAP --few-shot-shots 1  \
        --leaky --feature-processing ME --freeze-backbone  2>&1 | tee $SAVENAME;
    done
done


