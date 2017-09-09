import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import *
import numpy as np


def train_dur_epoch(model, dloader, opt, log_freq, epoch_idx,
                    cuda=False):
    model.train()
    epoch_losses = []
    num_batches = len(dloader)
    for b_idx, batch in enumerate(dloader):
        # decompose the batch into the sub-batches
        spk_b, lab_b, dur_b, slen_b = batch
        # convert all into variables and transpose (we want time-major)
        spk_b = Variable(spk_b).transpose(0,1).contiguous()
        lab_b = Variable(lab_b).transpose(0,1).contiguous()
        dur_b = Variable(dur_b).transpose(0,1).contiguous()
        slen_b = Variable(slen_b)
        # get curr batch size
        curr_bsz = spk_b.size(1)
        # init hidden states of dur model
        states = model.init_hidden_state(curr_bsz)
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
        loss = F.mse_loss(y.squeeze(-1), dur_b,
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
                   spk2durstats=None):
    model.eval()
    preds = None
    gtruths = None
    seqlens = None
    spks = None
    for b_idx, batch in enumerate(dloader):
        # decompose the batch into the sub-batches
        spk_b, lab_b, dur_b, slen_b = batch
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
            if preds is None:
                #print('Trimming seqlen {}/{}'.format(slen_i, y_i.shape[0]))
                preds = y_i[:slen_i]
                gtruths = dur_i[:slen_i]
                spks = spk_i[:slen_i]
            else:
                #print('Trimming seqlen {}/{}'.format(slen_i, y_i.shape[0]))
                preds = np.concatenate((preds, y_i[:slen_i]))
                gtruths = np.concatenate((gtruths, dur_i[:slen_i]))
                spks = np.concatenate((spks, spk_i[:slen_i]))
    if spk2durstats is not None:
        # denorm based on spk_id stats
        for ii, (spk_i, dur_i, y_i) in enumerate(zip(spks, gtruths, 
                                                     preds)):
            dur_stats = spk2durstats[spk_i]
            dur_min = dur_stats['min']
            dur_max = dur_stats['max']
            preds[ii] = denorm_minmax(y_i, dur_min, dur_max)
            gtruths[ii] = denorm_minmax(dur_i, dur_min, dur_max)
            #print('Denorming spk {} min: {}, max: {}'.format(spk_i, dur_min,
            #                                                 dur_max))
            #print('denorm {}-pred: {} s'.format(ii, preds[ii]))
            #print('denorm {}-gtruths: {} s'.format(ii, gtruths[ii]))
    dur_rmse = rmse(preds, gtruths) * 1e3
    print('Evaluated dur mRMSE [ms]: {:.3f}'.format(dur_rmse))
    return dur_rmse


