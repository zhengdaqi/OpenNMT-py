python  train.py -data /home/zhengdaqi/ws/attcnn/opennmt/data/ -save_model /home/zhengdaqi/ws/attcnn/opennmt/train/ -gpuid 3 \
        -layers 6 -rnn_size 512 -word_vec_size 512   \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -epochs 50  -max_generator_batches 32 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 \
        -use_attcnn 1
