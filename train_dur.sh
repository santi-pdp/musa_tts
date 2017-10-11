#!/bin/bash

SAVE_PATH="dur_experiments"

mkdir -p $SAVE_PATH

# train without trimming sequences and MSE
#python main.py --train-dur \
#	--save_path "$SAVE_PATH/ckpt_dur_mse_bsize15" --batch_size 15 --cuda --dur_loss mse \
#	--dur_lr 0.001 --cfg_spk cfg/tcstar.cfg --force-gen --exclude_train_spks 75 --exclude_eval_spks 75 \
#	--dur_max_samples 2

# train without trimming sequences and BCE
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_bce_bsize15" --batch_size 15 --cuda --dur_loss bce --sigmoid_dur \
#	--dur_lr 0.001 --exclude_train_spks 75 --exclude_eval_spks 75
#
## train without trimming sequences and k classes
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_k256_bsize15" --batch_size 15 --cuda \
#	--dur_lr 0.001 --dur_q_classes 256 --force-gen --exclude_train_spks 75 --exclude_eval_spks 75
#
# train trimming maxseqlen and BCE
python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_bce_maxseqlen35_bsize15" --batch_size 15 --cuda --dur_loss bce --sigmoid_dur  \
	--dur_max_seq_len 35 --dur_lr 0.001 --exclude_train_spks 75 --exclude_eval_spks 75 --dur_max_samples 100

## train trimming maxseqlen and k classes
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_k256_maxseqlen35_bsize15" --batch_size 15 --cuda --dur_q_classes 256 \
#	--dur_max_seq_len 35 --dur_lr 0.001 --exclude_train_spks 75 --exclude_eval_spks 75
#
## train only 72 speaker with BCE
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_72_bce_maxseqlen35_bsize15" --batch_size 15 --cuda --dur_loss bce --sigmoid_dur  \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg
#
## train only 73 speaker with BCE
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_73_bce_maxseqlen35_bsize15" --batch_size 15 --cuda --dur_loss bce --sigmoid_dur  \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_73.cfg
#
## train only 72 speaker with MSE
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_72_mse_maxseqlen35_bsize15" --batch_size 15 --cuda --dur_loss mse \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg
#
## train only 73 speaker with MSE
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_73_mse_maxseqlen35_bsize15" --batch_size 15 --cuda --dur_loss mse  \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_73.cfg
#
## train MO with 5 speakers (75 left apart)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_mo-no75_bsize15" --batch_size 15 --cuda \
#	--dur_lr 0.001 --dur_loss mse --force-gen --exclude_train_spks 75 --exclude_eval_spks 75 \
#	--dur_mulout
#
## train MO with 5 speakers (75 left apart)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_mo-no75_maxseqlen35_bsize15" --batch_size 15 --cuda \
#	--dur_lr 0.001 --dur_loss mse --dur_max_seq_len 35 --exclude_train_spks 75 --exclude_eval_spks 75 \
#	--dur_mulout
#
## train only 72 speaker, MSE and untrimmed
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_72_mse_bsize15" --batch_size 15 --cuda \
#	--dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg 
#
## ------------------------------------------
## BEWARE, as we regenerate codebooks, these quantification experiments have to be compared relatively to each other
## train only 72 speaker with k256 (have to re-generate quantization to optimize based on speaker)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_72_k256_maxseqlen35_bsize15" --batch_size 15 --cuda \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg --dur_q_classes 256 --force-gen
#
## train only 73 speaker with k256 (have to re-generate quantization to optimize based on speaker)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_73_k256_maxseqlen35_bsize15" --batch_size 15 --cuda \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_73.cfg --dur_q_classes 256 --force-gen
#
## train only 72 speaker with k128 (have to re-generate quantization to optimize based on speaker)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_72_k128_maxseqlen35_bsize15" --batch_size 15 --cuda \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg --dur_q_classes 128 --force-gen
#
## train only 72 speaker with k64 (have to re-generate quantization to optimize based on speaker)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_72_k64_maxseqlen35_bsize15" --batch_size 15 --cuda \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_72.cfg --dur_q_classes 64 --force-gen
#
## train only 73 speaker with k512 (have to re-generate quantization to optimize based on speaker)
#python main.py --train-dur --save_path "$SAVE_PATH/ckpt_dur_73_k512_maxseqlen35_bsize15" --batch_size 15 --cuda \
#	--dur_max_seq_len 35 --dur_lr 0.001 --cfg_spk cfg/tcstar_73.cfg --dur_q_classes 512 --force-gen
#
#
