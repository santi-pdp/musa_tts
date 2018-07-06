#!/bin/bash

#path='aco_ckpt_73'
#path='sattaco_N3_dff1024_embsz256_maxseqlen120_stPOSE_Noam_ckpt_73_pat20'
#path='rnn450_embsz128_aco_maxseqlen120_ckpt_73_pat20'
path='rnn1300_embsz512_aco_maxseqlen120_ckpt_73_pat20'
#lab='T6B72110000.lab'
#lab='T6B73200230.lab'
lab='T6B73200187.lab'
#epoch='11'
#epoch='18'
epoch='12'

python synthesize.py --model_cfg $path/main.opts \
	--aco_model $path/best-val_e"$epoch"_aco_model.ckpt \
	--synthesize_lab data/tcstar/lab/73/$lab --force-dur --cfg cfg/tcstar_73.cfg --pf 1.04  --cuda
