
touch log/attention.log
log='log/attention.log'

python ./attention_train.py     --train_data ./data/train_resnet50.txt \
                             --test_data ./data/test_resnet50.txt \
                             --model ./model/resnet50_iter_3000.pkl  \
                             --n_classes 17 \
                             --offset 1 \
                             --lr 0.1 \
                             --lr_step 500 \
                             --gamma 0.1 \
                             --batch_size 16 \
                             --test_batch_size 8 \
                             --test_iters 200 \
                             --log $log \
                             --save_iter 1000 \
                             --shuffle True \
                             --iters 3000 \
                             --output_dir ./model/ \
                             --display_iter 50  \


