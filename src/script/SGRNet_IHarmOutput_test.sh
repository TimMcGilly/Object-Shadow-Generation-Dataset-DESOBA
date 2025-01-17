
#!/bin/bash
batchs=1
GPU=0
lr=0.0002
loadSize=256
fineSize=256
L1=100
model=SGRNet
G='RESNEXT18'
ngf=32


L_shadowrecons=10
L_imagerecons=10
L_GAN=0.1

#####network design
DISPLAY_PORT=8002
D='pixel'
lr_D=0.0002


#####datset selected
datasetmode=iharmoutput


checkpoint='../../TrainedModels/SGRNet_TrainedModel'

#####dataroot is now IH output folder
dataroot='../../../IntrinsicHarmony/results/iih_base_lt_gd_allihd/test_latest/images'


model_name=SelfAttention
NAME="${model_name}_G${G}_C${ngf}_D${D}_lrD${lr_D}"

# TESTDATA="--bos"
TESTDATA="--bos"

OTHER="--no_crop --no_flip --no_rotate --serial_batches --no_dropout"

CMD="python ../test.py --loadSize ${loadSize} \
    --phase test --eval
    --name ${NAME} \
    --checkpoints_dir ${checkpoint} \
    --epoch latest\
    --fineSize $fineSize --model $model\
    --batch_size $batchs --display_port ${DISPLAY_PORT}
    --display_server http://localhost
    --gpu_ids ${GPU} --lr ${lr} \
    --dataset_mode $datasetmode\
    --norm instance\
    --dataroot  ${dataroot}\
    --lambda_M1 $L_shadowrecons --lambda_I1 $L_imagerecons --lambda_GAN $L_GAN 
    --netG $G\
    --ngf $ngf
    --netD $D
    --lr_D $lr_D

    $TESTDATA
    $OTHER"

echo $CMD
eval $CMD

