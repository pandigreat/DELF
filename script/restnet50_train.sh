
touch log/logresnet50.log
log='log/resnet50.log'

python ./train_resnet.py   --train_data ./data/train_resnet50.txt \
                             --test_data ./data/test_resnet50.txt \
                             --model ./model/   \
                             --n_classes 17 \
                             --offset 1 \
                             --lr 0.1 \
                             --lr_step 500 \
                             --gamma 0.1 \
                             --batch_size 32 \
                             --test_batch_size 4 \
                             --test_iters 200 \
                             --log $log \
                             --save_iter 1000 \
                             --shuffle True \
                             --iters 3000 \
                             --output_dir ./model/ \
                             --display_iter 100  \
