#!/bin/bash

if [ $# -lt 4 ]; then
	echo "Error, please specify: 1) lab_file, 2) save_path, 3) dur_weights, 4) aco_weights"
	exit 1;
fi
LAB_FILE="$1"
SAVE_PATH="$2"
DUR_WEIGHTS="$3"
ACO_WEIGHTS="$4"
ACO_DATA="$5"

python main.py --synthesize_lab $LAB_FILE --save_path $SAVE_PATH \
	--dur_weights $DUR_WEIGHTS --aco_weights $ACO_WEIGHTS --force-dur --pf 1
	#--aco_emb_size 256 --aco_rnn_layers 2 --aco_rnn_size 512  # --force-dur --aco_lab_norm znorm

LAB_FNAME=${LAB_FILE##*/}
LAB_BNAME=${LAB_FNAME%.*}
# Ahodecode wav file
ahodecoder16_64 "$SAVE_PATH/$LAB_BNAME".lf0 "$SAVE_PATH/$LAB_BNAME".mfcc "$SAVE_PATH/$LAB_BNAME".fv "$SAVE_PATH/$LAB_BNAME".wav 
#x2x +af "$ACO_DATA/$LAB_BNAME".cc > /tmp/lab.cc
#ahodecoder16_64 "$SAVE_PATH/$LAB_BNAME".lf0 /tmp/lab.cc "$SAVE_PATH/$LAB_BNAME".fv "$SAVE_PATH/$LAB_BNAME".wav 

