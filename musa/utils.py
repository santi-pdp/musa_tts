import torch
from torch.autograd import Variable
import numpy as np


def var_to_cuda(var):
    if isinstance(var, Variable):
        return var.cuda()
    elif isinstance(var, tuple): 
        return tuple(v.cuda() for v in var)
    elif isinstance(var, list):
        return [v.cuda() for v in var]
    else:
        raise TypeError('Incorrect var type to cuda')

def repackage_hidden(h, curr_bsz):
    """ Coming from https://github.com/pytorch/examples/blob/master/word_language_model/main.py """
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data[:, :curr_bsz, :]).contiguous()
    else:
        return tuple(repackage_hidden(v, curr_bsz).contiguous() for v in h)


def rmse(prediction, groundtruth):
    assert prediction.shape == groundtruth.shape
    D = np.sqrt(np.mean((groundtruth - prediction) ** 2, axis=0))
    return D

def denorm_minmax(y, out_min, out_max):
    # x = y * (max - min) + min
    R = out_max - out_min
    x = y * R + out_min
    #print('denorm minmax {} -> {}'.format(y, x))
    return y * R + out_min

def predict_masked_rmse(y, dur_b, slen_b, spk_b, curr_ph_b,
                        preds, gtruths, spks, sil_mask,
                        sil_id,
                        q_classes):
    y_npy = y.cpu().data.transpose(0,1).numpy()
    dur_npy = dur_b.cpu().data.transpose(0,1).numpy()
    #print('dur_npy max: ', dur_npy.max())
    #print('dur_npy.shape: ', dur_npy.shape)
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
            if q_classes:
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
            if q_classes:
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
    return preds, gtruths, spks, sil_mask

def denorm_dur_preds_gtruth(preds, gtruths, spks, spk2durstats,
                            q_classes):
    # denorm based on spk_id stats
    for ii, (spk_i, dur_i, y_i) in enumerate(zip(spks, gtruths, 
                                                 preds)):
        dur_stats = spk2durstats[spk_i]
        #print('selected dur_stats {} for spk {}'.format(dur_stats, spk_i))
        if q_classes:
            kmeans = dur_stats
            # map gtruths idxes to centroid values
            ccs = kmeans.cluster_centers_
            dur_cc = ccs[int(dur_i)][0]
            #print('Groundtruth cc {} from dur {}'.format(dur_cc,
            #                                             dur_i))
            gtruths[ii] = dur_cc
            # get max of predictions
            #print('Argmax pred: ', y_i)
            pred_cc = ccs[int(y_i)]
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
    return preds, gtruths

