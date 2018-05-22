python  train.py -data /home/zhengdaqi/ws/attcnn/opennmt/mfd.data.pt/wmt_en-de \
        -save_model /hd4T/zdq/ws/opennmt.m2m.run4.fix/ \
        -gpuid 3 \
        -layers 6 -rnn_size 512 -word_vec_size 512   \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -epochs 70  -max_generator_batches 32 -dropout 0.1 \
        -batch_size 2048 -batch_type tokens -normalization tokens  -accum_count 4 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 \
        -use_attcnn 2 \
        -use_pretrain 1 \
        -train_from /hd4T/zdq/ws/opennmt.baseline.2018.05.07/cur/_acc_68.19_ppl_4.82_e44.pt
