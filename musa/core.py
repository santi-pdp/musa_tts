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
    tr_loss = []
    va_loss = {}
    min_va_loss = np.inf
    patience=eval_patience
    for epoch in range(epochs):
        best_model = False
        tr_loss += train_fn(model, dloader, opt, log_freq, epoch,
                            criterion=train_criterion,
                            cuda=cuda, tr_opts=tr_opts.copy())
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
        print('Saving training loss')
        np.save(os.path.join(save_path, 'tr_loss'), tr_loss)
        if eval_target:
            for k, v in va_loss.items():
                print('Saving val score ', k)
                np.save(os.path.join(save_path, k), v)

def train_dur_epoch(model, dloader, opt, log_freq, epoch_idx,
                    criterion=None, cuda=False, tr_opts={}):
    model.train()
    stateful = False
    if 'stateful' in tr_opts:
        stateful = True
        tr_opts.pop('stateful')
    assert len(tr_opts) == 0, 'unrecognized params passed in: '\
                              '{}'.format(kwargs.keys())
    epoch_losses = []
    num_batches = len(dloader)
    for b_idx, batch in enumerate(dloader):
        # decompose the batch into the sub-batches
        spk_b, lab_b, dur_b, slen_b, _ = batch
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
        #print('y size: ', y.size())
        #print('states[0] size: ', states[0].size())
        # compute loss
        if criterion == F.nll_loss:
            y = y.view(-1, y.size(-1))
            dur_b = dur_b.view(-1)
        #print('y size: ', y.size())
        #print('dur_b: ', dur_b.size())
        loss = criterion(y.squeeze(-1), dur_b,
                         size_average=True)
        #print('batch {:4d}: loss: {:.5f}'.format(b_idx + 1, loss.data[0]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (b_idx + 1) % log_freq == 0 or (b_idx + 1) >= num_batches:
            print('batch {:4d}/{:4d} (epoch {:3d}) loss '
                  '{:.5f}'.format(b_idx + 1, num_batches, epoch_idx,
                                  loss.data[0]))
            epoch_losses.append(loss.data[0])
    print('-- Finished epoch {:4d}, mean tr loss: '
          '{:.5f}'.format(epoch_idx, np.mean(epoch_losses)))
    return epoch_losses

def eval_dur_epoch(model, dloader, epoch_idx, cuda=False,
                   stats=None, va_opts={'sil_id':'pau'}):
    model.eval()
    sil_id = 'pau'
    q_classes = False
    if 'sil_id' in va_opts:
        sil_id = va_opts.pop('sil_id')
    if 'q_classes' in va_opts:
        q_classes = True
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
        y = y.squeeze(-1)
        y_npy = y.cpu().data.transpose(0,1).numpy()
        dur_npy = dur_b.cpu().data.transpose(0,1).numpy()
        slens_npy = slen_b.cpu().data.numpy()
        spk_npy = spk_b.cpu().data.transpose(0,1).numpy()
        # first, select sequences within permitted lengths (remove pad)
        for ii, (y_i, dur_i, spk_i, slen_i) in enumerate(zip(y_npy, 
                                                             dur_npy, 
                                                             spk_npy, 
                                                             slens_npy)):
            # get curr phoneme identity
            curr_ph_seq = curr_ph_b[ii]
            # create seq_mask
            curr_ph_seq_mask = np.zeros((slen_i,))
            for t_ in range(slen_i):
                curr_ph = curr_ph_seq[t_]
                if curr_ph != sil_id:
                    curr_ph_seq_mask[t_] = 1.
                else:
                    curr_ph_seq_mask[t_] = 0.
            if preds is None:
                #print('Trimming seqlen {}/{}'.format(slen_i, y_i.shape[0]))
                if q_classes is not None:
                    max_y_i = []
                    # have to argmax directly y_i seq
                    for y_i_t in y_i:
                        max_y_i.append(np.argmax(y_i_t))
                    preds = np.array(max_y_i[:slen_i], dtype=np.float32)
                else:
                    # just single output y_i_t
                    preds = np.array(y_i[:slen_i], dtype=np.float32)
                gtruths = np.array(dur_i[:slen_i], dtype=np.float32)
                spks = spk_i[:slen_i]
                sil_mask = curr_ph_seq_mask
            else:
                #print('Trimming seqlen {}/{}'.format(slen_i, y_i.shape[0]))
                if q_classes is not None:
                    max_y_i = []
                    # have to argmax directly y_i seq
                    for y_i_t in y_i:
                        max_y_i.append(np.argmax(y_i_t))
                    preds = np.concatenate((preds, np.array(max_y_i[:slen_i], 
                                                            dtype=np.float32)))
                else:
                    preds = np.concatenate((preds, np.array(y_i[:slen_i],
                                                            dtype=np.float32)))
                gtruths = np.concatenate((gtruths, np.array(dur_i[:slen_i],
                                                            dtype=np.float32)))
                spks = np.concatenate((spks, spk_i[:slen_i]))
                #print('concatenating sil_mask shape: ', sil_mask.shape)
                #print('with curr_ph_seq_mask shape: ', curr_ph_seq_mask.shape)
                sil_mask = np.concatenate((sil_mask, curr_ph_seq_mask))
    #print('After batch preds shape: ', preds.shape)
    #print('After batch gtruths shape: ', gtruths.shape)
    #print('After batch sil_mask shape: ', sil_mask.shape)
    # denorm with normalization stats
    assert spk2durstats is not None
    # denorm based on spk_id stats
    for ii, (spk_i, dur_i, y_i) in enumerate(zip(spks, gtruths, 
                                                 preds)):
        dur_stats = spk2durstats[spk_i]
        if not q_classes:
            kmeans = dur_stats#['dur_clusters']
            # map gtruths idxes to centroid values
            ccs = kmeans.cluster_centers_
            dur_cc = ccs[dur_i][0]
            #print('Groundtruth cc {} from dur {}'.format(dur_cc,
            #                                             dur_i))
            gtruths[ii] = dur_cc
            # get max of predictions
            #print('Argmax pred: ', y_i)
            pred_cc = ccs[y_i]
            #print('Prediction cc {} from dur {}'.format(pred_cc,
            #                                            y_i))
            preds[ii] = pred_cc
            #print('gtruth: {} s'.format(gtruths[ii]))
            #print('pred: {} s'.format(preds[ii]))
        else:
            dur_min = dur_stats['min']
            dur_max = dur_stats['max']
            preds[ii] = denorm_minmax(y_i, dur_min, dur_max)
            gtruths[ii] = denorm_minmax(dur_i, dur_min, dur_max)
        #print('Denorming spk {} min: {}, max: {}'.format(spk_i, dur_min,
        #                                                 dur_max))
        #print('denorm {}-pred: {} s'.format(ii, preds[ii]))
        #print('denorm {}-gtruths: {} s'.format(ii, gtruths[ii]))
        # decode with kmeans object
        kmeans = spk2durstats[spk_i]
    dur_rmse = rmse(preds, gtruths) * 1e3
    nosil_dur_rmse = rmse(preds * sil_mask, gtruths * sil_mask) * 1e3
    print('Evaluated dur mRMSE [ms]: {:.3f}'.format(dur_rmse))
    print('Evaluated dur w/o sil phones mRMSE [ms]:'
          '{:.3f}'.format(nosil_dur_rmse))
    # TODO: Make dur evaluation speaker-wise
    return {'total_dur_rmse':dur_rmse, 'total_nosil_dur_rmse':nosil_dur_rmse}


