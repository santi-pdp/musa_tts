#!/bin/bash

# MO acoustic
python main.py --train-aco --save_path test_mo --exclude_train_spks 75 \
     	       --aco_max_seq_len 35 --exclude_eval_spks 75 \
               --batch_size 15 --aco_mulout --force-gen --cuda

# Train each speaker 72 and 73

python main.py --train-aco --save_path so_aco_1spk-72 --batch_size 15 \
	--cuda --aco_max_seq_len 35 --batch_size 15 --cfg cfg/tcstar_72.cfg \
	--aco_train_forced_trim 17879 --aco_valid_forced_trim 2294


python main.py --train-aco --save_path so_aco_1spk-73 --batch_size 15 \
	--cuda --aco_max_seq_len 35 --batch_size 15 --cfg cfg/tcstar_73.cfg \
	--aco_train_forced_trim 17879 --aco_valid_forced_trim 2294

	

# SO acoustic
#python main.py --train-aco --save_path so_aco_5spk --exclude_train_spks 75 \
#	      --aco_max_seq_len 35 --exclude_eval_spks 75 \
#              --batch_size 15  
