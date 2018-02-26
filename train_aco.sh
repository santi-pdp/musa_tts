#!/bin/bash

SAVE_PATH="aco_experiments"
BATCH_SIZE=50
SEQ_LEN=50

# SO acoustic (73)
python -u main.py --train-aco --save_path "$SAVE_PATH/"so_aco_73 \
		  --aco_max_seq_len $SEQ_LEN \
		  --batch_size $BATCH_SIZE --cuda --cfg cfg/tcstar_73.cfg

# MO acoustic (5 spk)
python -u main.py --train-aco --save_path "$SAVE_PATH/"mo_aco_5spk \
		  --aco_max_seq_len $SEQ_LEN \
		  --batch_size $BATCH_SIZE --cuda --cfg cfg/tcstar.cfg \
		  --exclude_train_spks 75 --exclude_eval_spks 75 --aco_mulout

# SO acoustic (5 spk)
python -u main.py --train-aco --save_path "$SAVE_PATH/"so_aco_5spk --exclude_train_spks 75 \
		  --aco_max_seq_len $SEQ_LEN --exclude_eval_spks 75 \
		  --batch_size $BATCH_SIZE --cuda 
