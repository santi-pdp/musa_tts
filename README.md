# MUSA Text-to-speech

### Synthesize

With forced duration read from label file:
```
python synthesize.py --model_cfg aco_73_ckpt/main.opts \
        --aco_model aco_73_ckpt/best-val_e{epoch}_aco_model.ckpt \
        --synthesize_lab <lab_path>.lab --force-dur --cfg cfg/tcstar_73.cfg --cuda --pf 1.04
```

With prediction coming from duration model:
```
python synthesize.py --model_cfg aco_73_ckpt/main.opts \
        --aco_model aco_73_ckpt/best-val_e{epoch}_aco_model.ckpt \
        --dur_model aco_73_ckpt/best-val_e{epoch}_dur_model.ckpt \
        --synthesize_lab <lab_path>.lab --cfg cfg/tcstar_73.cfg --cuda --pf 1.04
```

### Train the acoustic model

```
python train_aco.py --save_path aco_73_ckpt --cuda --cfg cfg/tcstar_73.cfg --batch_size 32 --epoch 100 --patience 20 --max_seq_len 50 
```

### Train the duration model
```
python train_dur.py --save_path dur_73_ckpt --cuda --cfg cfg/tcstar_73.cfg --batch_size 32 --epoch 100 --patience 20 --max_seq_len 50 
```

### TODO:

* Include instrunctions on how to use our latest SALAD model in this README.
