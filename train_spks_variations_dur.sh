#!/bin/bash

# 72 speaker
python main.py --train-dur --save_path ckpt_dur_bce_72_maxseqlen --batch_size 15 --cuda --dur_loss bce \
	       --sigmoid_dur  --dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg 

# 73 speaker
python main.py --train-dur --save_path ckpt_dur_bce_73_maxseqlen --batch_size 15 --cuda --dur_loss bce \
	       --sigmoid_dur  --dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_73.cfg 
