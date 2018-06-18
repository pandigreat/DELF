
touch log/logresnet50.log
log='log/resnet50.log'

python ./train_resnet.py   --train_data ./data/train_resnet50.txt \
                             --test_data ./data/test_resnet50.txt \
                             --model ./model/   \
                             --nclasses 17 \
                             --offset 1 \
                             --lr 0.1 \
                             --lr_step 500 \
                             --gamma 0.1 \
                             --batch_size 32 \
                             --test_batch_size 8 \
                             --test_iter 200 \
                             --log $log \
                             --save_iter 1000 \
                             --shuffle True \
                             --iters 3000 \
