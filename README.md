# musa_tts

### Synthesize

```
bash synthesize.sh data/tcstar/lab/72/T6B72110000.lab 73_testing \
     dur_experiments/ckpt_dur_73_mse_maxseqlen35_bsize15/best-val_e12_dur_model.ckpt MUSA/l1_so-73_aco_fulldata/best-val_e15_aco_model.ckpt
```

### Train

```
python main.py --train-aco --save_path MUSA/so-73_aco_fulldata_20patience --aco_max_seq_len 35 --batch_size 50 --cuda --cfg cfg/tcstar_73.cfg --patience 20
```
