import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import *
import numpy as np
import os


def train_engine(model, dloader, opt, log_freq, train_fn, train_criterion,
                 epochs, save_path, model_savename, tr_opts={}, eval_fn=None, 
                 val_dloader=None, eval_stats=None, eval_target=None, 
                 eval_patience=None, cuda=False, va_opts={}):
    tr_loss = {}
    va_loss = {}
    min_va_loss = np.inf
    patience=eval_patience
    for epoch in range(epochs):
        best_model = False
        tr_e_loss = train_fn(model, dloader, opt, log_freq, epoch,
                             criterion=train_criterion,
                             cuda=cuda, tr_opts=tr_opts.copy())
        for k, v in tr_e_loss.items():
            if k not in tr_loss:
                tr_loss[k] = [v]
            else:
                tr_loss[k].append(v)
        if eval_fn:
            if val_dloader is None:
                raise ValueError('Train engine: please specify '
                                 'a validation data loader!')
            val_scores = eval_fn(model, val_dloader, 
                                 epoch, cuda=cuda,
                                 stats=eval_stats,
                                 va_opts=va_opts.copy())
            if eval_target:
                if eval_patience is None:
                    raise ValueError('Train engine: Need a patience '
                                     'factor to be specified '
                                     'whem eval_target is given')
                for k, v in val_scores.items():
                    if k not in va_loss:
                        va_loss[k] = [v]
                    else:
                        va_loss[k].append(v)
                # we have a target key to do early stopping upon it
                if val_scores[eval_target] < min_va_loss:
                    print('Val loss improved {:.3f} -> {:.3f}'
                          ''.format(min_va_loss, val_scores[eval_target]))
                    min_va_loss = val_scores[eval_target]
                    best_model = True
                    patience = eval_patience
                else:
                    patience -= 1
                    print('Val loss did not improve. Curr '
                          'patience: {}/{}'.format(patience,
                                                   eval_patience))
                    if patience == 0:
                        print('Out of patience. Ending DUR training.')
                        break
        model.save(save_path, model_savename, epoch,
                   best_val=best_model)
        for k, v in tr_loss.items():
            print('Saving training loss ', k)
            np.save(os.path.join(save_path, k), v)
        if eval_target:
            for k, v in va_loss.items():
                print('Saving val score ', k)
                np.save(os.path.join(save_path, k), v)

def train_dur_epoch(model, dloader, opt, log_freq, epoch_idx,
                    criterion=None, cuda=False, tr_opts={},
                    spk2durstats=None):
    model.train()
    stateful = False
    if 'stateful' in tr_opts:
        stateful = True
        tr_opts.pop('stateful')
    spk2durstats = None
    if 'spk2durstats' in tr_opts:
        print('Getting spk2durstats')
        spk2durstats = tr_opts.pop('spk2durstats')
    mulout = False
    if 'mulout' in tr_opts:
        print('Multi-Output dur training')
        mulout = tr_opts.pop('mulout')
        if 'idx2spk' in tr_opts:
            idx2spk = tr_opts.pop('idx2spk')
        else:
            raise ValueError('Specify a idx2spk in training opts '
                             'when using MO.')
    assert len(tr_opts) == 0, 'unrecognized params passed in: '\
                              '{}'.format(tr_opts.keys())
    epoch_losses = {}
    num_batches = len(dloader)
    for b_idx, batch in enumerate(dloader):
        # decompose the batch into the sub-batches
        spk_b, lab_b, dur_b, slen_b, ph_b = batch
        # build batch of curr_ph to filter out results without sil phones
        # size of curr_ph_b [bsize, seqlen]
        curr_ph_b = [[ph[2] for ph in ph_s] for ph_s in ph_b]
        # convert all into variables and transpose (we want time-major)
        spk_b = Variable(spk_b).transpose(0,1).contiguous()
        lab_b = Variable(lab_b).transpose(0,1).contiguous()
        dur_b = Variable(dur_b).transpose(0,1).contiguous()
        slen_b = Variable(slen_b)
        # get curr batch size
        curr_bsz = spk_b.size(1)
        if (stateful and b_idx == 0) or not stateful:
            #print('Initializing recurrent states, e: {}, b: '
            #      '{}'.format(epoch_idx, b_idx))
            # init hidden states of dur model
            states = model.init_hidden_state(curr_bsz)
        if stateful and b_idx > 0:
            #print('Copying recurrent states, e: {}, b: '
            #      '{}'.format(epoch_idx, b_idx))
            # copy last states
            states = repackage_hidden(states, curr_bsz)
        if cuda:
            spk_b = var_to_cuda(spk_b)
            lab_b = var_to_cuda(lab_b)
            dur_b = var_to_cuda(dur_b)
            slen_b = var_to_cuda(slen_b)
            states = var_to_cuda(states)
        # forward through model
        y, states = model(lab_b, states, speaker_idx=spk_b)
        if isinstance(y, dict):
            # we have a MO model, pick the right spk
            spk_name = idx2spk[spk_b.cpu().data[0,0]]
            # print('Extracting y prediction for MO spk ', spk_name)
            y = y[spk_name]
        q_classes = False
        #print('y size: ', y.size())
        #print('states[0] size: ', states[0].size())
        # compute loss
        if criterion == F.nll_loss:
            y = y.view(-1, y.size(-1))
            dur_b = dur_b.view(-1)
            q_classes = True
        y = y.squeeze(-1)
        preds = None
        gtruths = None
        seqlens = None
        spks = None
        # make the silence mask
        sil_mask = None
        preds, gtruths, \
        spks, sil_mask = predict_masked_rmse(y, dur_b, slen_b, 
                                             spk_b, curr_ph_b,
                                             preds, gtruths,
                                             spks, sil_mask,
                                             'pau',
                                             q_classes)
        #print('Tr After batch preds shape: ', preds.shape)
        #print('Tr After batch gtruths shape: ', gtruths.shape)
        #print('Tr After batch sil_mask shape: ', sil_mask.shape)
        # denorm with normalization stats
        assert spk2durstats is not None
        preds, gtruths = denorm_dur_preds_gtruth(preds, gtruths,
                                                 spks, spk2durstats,
                                                 q_classes)
        #print('preds[:20] = ', preds[:20])
        #print('gtruths[:20] = ', gtruths[:20])
        #print('pred min: {}, max: {}'.format(preds.min(), preds.max()))
        #print('gtruths min: {}, max: {}'.format(gtruths.min(), gtruths.max()))
        nosil_dur_rmse = rmse(preds * sil_mask, gtruths * sil_mask) * 1e3
        #print('y size: ', y.size())
        #print('dur_b: ', dur_b.size())
        loss = criterion(y, dur_b,
                         size_average=True)
        #print('batch {:4d}: loss: {:.5f}'.format(b_idx + 1, loss.data[0]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        #print('y size: ', y.size())
        if (b_idx + 1) % log_freq == 0 or (b_idx + 1) >= num_batches:

            print('batch {:4d}/{:4d} (epoch {:3d}) loss '
                  '{:.5f}, rmse {:.5f}ms'.format(b_idx + 1, 
                                                 num_batches, 
                                                 epoch_idx,
                                                 loss.data[0],
                                                 nosil_dur_rmse))
            if 'tr_loss' not in epoch_losses:
                epoch_losses['tr_loss'] = [loss.data[0]]
                epoch_losses['tr_rmse'] = [nosil_dur_rmse]
            else:
                epoch_losses['tr_loss'].append(loss.data[0])
                epoch_losses['tr_rmse'].append(nosil_dur_rmse)
            #epoch_losses.append(loss.data[0])
    print('-- Finished epoch {:4d}, mean tr loss: '
          '{:.5f}'.format(epoch_idx, np.mean(epoch_losses['tr_loss'])))
    return epoch_losses

def eval_dur_epoch(model, dloader, epoch_idx, cuda=False,
                   stats=None, va_opts={}):
    model.eval()
    sil_id = 'pau'
    q_classes = False
    if 'sil_id' in va_opts:
        sil_id = va_opts.pop('sil_id')
    if 'q_classes' in va_opts:
        q_classes= va_opts.pop('q_classes')
    if 'mulout' in va_opts:
        print('Multi-Output dur evaluation')
        mulout = va_opts.pop('mulout')
        if 'idx2spk' in va_opts:
            idx2spk = va_opts.pop('idx2spk')
        else:
            raise ValueError('Specify a idx2spk in eval opts '
                             'when using MO.')
    assert len(va_opts) == 0, 'unrecognized params passed in: '\
                              '{}'.format(va_opts.keys())
    spk2durstats=stats
    preds = None
    gtruths = None
    seqlens = None
    spks = None
    # make the silence mask
    sil_mask = None
    for b_idx, batch in enumerate(dloader):
        # decompose the batch into the sub-batches
        spk_b, lab_b, dur_b, slen_b, ph_b = batch
        # build batch of curr_ph to filter out results without sil phones
        # size of curr_ph_b [bsize, seqlen]
        curr_ph_b = [[ph[2] for ph in ph_s] for ph_s in ph_b]
        # convert all into variables and transpose (we want time-major)
        spk_b = Variable(spk_b, volatile=True).transpose(0,1).contiguous()
        lab_b = Variable(lab_b, volatile=True).transpose(0,1).contiguous()
        dur_b = Variable(dur_b, volatile=True).transpose(0,1).contiguous()
        slen_b = Variable(slen_b, volatile=True)
        # get curr batch size
        curr_bsz = spk_b.size(1)
        # init hidden states of dur model
        states = model.init_hidden_state(curr_bsz, volatile=True)
        if cuda:
            spk_b = var_to_cuda(spk_b)
            lab_b = var_to_cuda(lab_b)
            dur_b = var_to_cuda(dur_b)
            slen_b = var_to_cuda(slen_b)
            states = var_to_cuda(states)
        # forward through model
        y, states = model(lab_b, states, speaker_idx=spk_b)
        if isinstance(y, dict):
            # we have a MO model, pick the right spk
            spk_name = idx2spk[spk_b.cpu().data[0,0]]
            # print('Extracting y prediction for MO spk ', spk_name)
            y = y[spk_name]
        y = y.squeeze(-1)
        preds, gtruths, \
        spks, sil_mask = predict_masked_rmse(y, dur_b, slen_b, 
                                             spk_b, curr_ph_b,
                                             preds, gtruths,
                                             spks, sil_mask,
                                             sil_id,
                                             q_classes)
    #print('After batch preds shape: ', preds.shape)
    #print('After batch gtruths shape: ', gtruths.shape)
    #print('After batch sil_mask shape: ', sil_mask.shape)
    # denorm with normalization stats
    assert spk2durstats is not None
    preds, gtruths = denorm_dur_preds_gtruth(preds, gtruths,
                                             spks, spk2durstats,
                                             q_classes)
    dur_rmse = rmse(preds, gtruths) * 1e3
    nosil_dur_rmse = rmse(preds * sil_mask, gtruths * sil_mask) * 1e3
    print('Evaluated dur mRMSE [ms]: {:.3f}'.format(dur_rmse))
    print('Evaluated dur w/o sil phones mRMSE [ms]:'
          '{:.3f}'.format(nosil_dur_rmse))
    # TODO: Make dur evaluation speaker-wise
    return {'total_dur_rmse':dur_rmse, 'total_nosil_dur_rmse':nosil_dur_rmse}


