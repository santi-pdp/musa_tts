#!/bin/bash

SAVE_PATH="aco_experiments"
BATCH_SIZE=50
SEQ_LEN=50
MEM=180000

# MO acoustic
srun -p veu --mem $MEM --gres=gpu:1 python main.py --train-aco --save_path "$SAVE_PATH/"mo_aco_5spk \
						    --exclude_train_spks 75 \
						    --aco_max_seq_len $SEQ_LEN --exclude_eval_spks 75 \
						    --batch_size $BATCH_SIZE --aco_mulout --force-gen --cuda | tee "$SAVE_PATH"/mo_aco_5spk.log &

# Train each speaker 72 and 73

# 72
#srun -p veu --mem $MEM --gres=gpu:1 python main.py --train-aco --save_path "$SAVE_PATH/"so_aco_1spk-72 \
#						    --aco_max_seq_len $SEQ_LEN --batch_size $BATCH_SIZE \
#						    --cuda --cfg cfg/tcstar_73.cfg \
#						    --aco_train_forced_trim 17879 --aco_valid_forced_trim 2294 | tee "$SAVE_PATH"/so_aco_1spk-72.log  &


# 73
#srun -p veu --mem $MEM --gres=gpu:1 python main.py --train-aco --save_path "$SAVE_PATH/"so_aco_1spk-73 \
#						    --aco_max_seq_len $SEQ_LEN --batch_size $BATCH_SIZE \
#						    --cuda --cfg cfg/tcstar_72.cfg \
#						    --aco_train_forced_trim 17879 --aco_valid_forced_trim 2294 | tee "$SAVE_PATH"/so_aco_1spk-72.log  &


# SO acoustic
srun -p veu --mem $MEM --gres=gpu:1 python main.py --train-aco --save_path "$SAVE_PATH/"so_aco_5spk --exclude_train_spks 75 \
						    --aco_max_seq_len $SEQ_LEN --exclude_eval_spks 75 \
						    --batch_size $BATCH_SIZE --cuda | tee "$SAVE_PATH"/so_aco_5spk.log  &
