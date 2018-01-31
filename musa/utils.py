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
    elif isinstance(var, dict):
        for k, el in var.items():
            var[k] = var_to_cuda(el)
        return var
    else:
        raise TypeError('Incorrect var type to cuda')

def repackage_hidden(h, curr_bsz):
    """ Coming from https://github.com/pytorch/examples/blob/master/word_language_model/main.py """
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data[:, :curr_bsz, :]).contiguous()
    elif isinstance(h, dict):
        # go element by element, repackaging
        for k, el in h.items():
            h[k] = repackage_hidden(el, curr_bsz)
        return h
    else:
        return tuple(repackage_hidden(v, curr_bsz).contiguous() for v in h)

def write_scalar_log(val, tag, step, log_writer=None):
    if log_writer is not None:
        log_writer.add_scalar(tag, val, step)

def write_histogram_log(val, tag, step, log_writer=None):
    if log_writer is not None:
        log_writer.add_histogram(tag, val, step, bins='sturges')


def rmse(prediction, groundtruth, spks=None, idx2spk=None):
    assert prediction.shape == groundtruth.shape
    # global
    D = np.asscalar(np.sqrt(np.mean((groundtruth - prediction) ** 2, axis=0)))
    if spks is not None:
        spk_durs = {}
        for (pred, gtruth, spk) in zip(prediction, groundtruth,
                                       spks):
            if str(spk) not in spk_durs:
                spk_durs[str(spk)] = []
            spk_durs[str(spk)].append((gtruth - pred) ** 2)
        spks = (spk_durs.keys())
        for spk in spks:
            diffs = spk_durs[spk]
            avg = np.mean(diffs, axis=0)
            spk_durs[spk] = np.asscalar(np.sqrt(avg))
        # remake dict if idx2spk available
        if idx2spk is not None:
            nspk_durs = {}
            for spk in spks:
                nspk_durs[idx2spk[int(spk)]] = spk_durs[spk]
            return D, nspk_durs
        return D, spk_durs
    else:
        return D

def accuracy(prediction, groundtruth):
    a = [not x for x in np.logical_xor(prediction, 
                                       groundtruth)]
    a = list(map(float, a))
    return np.sum(a) / len(a)

def fpr(prediction, groundtruth):
    """ Compute F-measure, Precision and Recall """
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(groundtruth, prediction)
    r = recall_score(groundtruth, prediction)
    f = f1_score(groundtruth, prediction)
    return f, p, r

def afpr(prediction, groundtruth, spks=None, idx2spk=None):
    assert prediction.shape == groundtruth.shape
    #print('afpr prediction shape: ', prediction.shape)
    #print('min pred uv: ', np.min(prediction))
    #print('max pred uv: ', np.max(prediction))
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
        groundtruth = groundtruth.reshape(-1, 1)
    if spks is not None:
        # recursively call afpr for each speaker
        spk_uvs = {}
        spk_res = {}
        for (pred, gtruth, spk) in zip(prediction, groundtruth,
                                       spks):
            if str(spk) not in spk_uvs:
                spk_uvs[str(spk)] = {'preds':[], 'gtruths':[]}
            spk_uvs[str(spk)]['preds'].append(pred)
            spk_uvs[str(spk)]['gtruths'].append(gtruth)
        spks = (spk_uvs.keys())
        #print('Eval mcd spks: ', spks)
        for spk in spks:
            spk_pred = np.array(spk_uvs[spk]['preds'])
            spk_gtruth = np.array(spk_uvs[spk]['gtruths'])
            spk_r = afpr(spk_pred, spk_gtruth)
            for k, v in spk_r.items():
                if idx2spk is not None:
                    spk_res['{}.{}'.format(k, idx2spk[int(spk)])] = v
                else:
                    spk_res['{}.{}'.format(k, spk)] = v
        # compute global too
        #print('Computing global afpr')
        total_res = afpr(prediction, groundtruth)
        spk_res['total'] = {}
        for k, v in total_res.items():
            spk_res['{}.total'.format(k)] = v
            spk_res['total']['{}.total'.format(k)] = v
        return spk_res
    else:
        #print('groundtruth.shape: ', groundtruth.shape)
        #print('prediction.shape: ', prediction.shape)
        f, p, r = fpr(prediction, groundtruth) 
        return {'A':accuracy(prediction, groundtruth),
                'F':f, 'P':p, 'R':r}

def mcd(prediction, groundtruth, spks=None, idx2spk=None):
    """ Mean Cepstral Distortion 
        Inputs are matrices of shape (time, cc_order)
    """
    assert prediction.shape == groundtruth.shape
    if spks is not None:
        # recursively call mcd for each speaker
        spk_ccs = {}
        spk_res = {}
        for (pred, gtruth, spk) in zip(prediction, groundtruth,
                                       spks):
            if str(spk) not in spk_ccs:
                spk_ccs[str(spk)] = {'preds':[], 'gtruths':[]}
            spk_ccs[str(spk)]['preds'].append(pred)
            spk_ccs[str(spk)]['gtruths'].append(gtruth)
        spks = (spk_ccs.keys())
        #print('Eval mcd spks: ', spks)
        for spk in spks:
            spk_pred = np.array(spk_ccs[spk]['preds'])
            spk_gtruth = np.array(spk_ccs[spk]['gtruths'])
            if idx2spk is not None:
                spk_res[idx2spk[int(spk)]] = mcd(spk_pred, spk_gtruth)
            else:
                spk_res[spk] = mcd(spk_pred, spk_gtruth)
        # compute global too
        spk_res['total'] = mcd(prediction, groundtruth)
        return spk_res
    else:
        #print('groundtruth.shape: ', groundtruth.shape)
        #print('prediction.shape: ', prediction.shape)
        mcd_ = 0
        for t in range(groundtruth.shape[0]):
            acum = 0
            for n in range(groundtruth.shape[1]):
                acum += (groundtruth[t, n] - prediction[t, n]) ** 2
            mcd_ += np.sqrt(acum)
        # scale factor
        alpha = ((10. * np.sqrt(2)) / (groundtruth.shape[0] * np.log(10)))
        mcd_ = alpha * mcd_
        return mcd_

def denorm_minmax(y, out_min, out_max):
    # x = y * (max - min) + min
    R = out_max - out_min
    x = y * R + out_min
    #print('denorm minmax {} -> {}'.format(y, x))
    return x

def predict_masked_mcd(y, aco_b, slen_b, spk_b, curr_ph_b,
                       preds, gtruths, spks, sil_mask,
                       sil_id):
    y_npy = y.cpu().data.transpose(0,1).numpy()
    aco_npy = aco_b.cpu().data.transpose(0,1).numpy()
    slens_npy = slen_b.cpu().data.numpy()
    spk_npy = spk_b.cpu().data.transpose(0,1).numpy()
    # first, select sequences within permitted lengths (remove pad)
    for ii, (y_i, aco_i, spk_i, slen_i) in enumerate(zip(y_npy, 
                                                         aco_npy, 
                                                         spk_npy, 
                                                         slens_npy)):
        # get curr phoneme identity
        curr_ph_seq = curr_ph_b[ii]
        # create seq_mask
        curr_ph_seq_mask = np.zeros((slen_i,))
        #print('curr_ph_seq: ', curr_ph_seq)
        for t_ in range(slen_i):
            curr_ph = curr_ph_seq[t_]
            if curr_ph != sil_id:
                curr_ph_seq_mask[t_] = 1.
            else:
                curr_ph_seq_mask[t_] = 0.
        #print('resulting mask: ', curr_ph_seq_mask)
        if preds is None:
            #print('Trimming seqlen {}/{}'.format(slen_i, y_i.shape[0]))
            preds = np.array(y_i[:slen_i], dtype=np.float32)
            gtruths = np.array(aco_i[:slen_i], dtype=np.float32)
            spks = spk_i[:slen_i]
            sil_mask = curr_ph_seq_mask.reshape((-1, 1))
        else:
            #print('Trimming seqlen {}/{}'.format(slen_i, y_i.shape[0]))
            preds = np.concatenate((preds, np.array(y_i[:slen_i],
                                                    dtype=np.float32)))
            gtruths = np.concatenate((gtruths, np.array(aco_i[:slen_i],
                                                        dtype=np.float32)))
            spks = np.concatenate((spks, spk_i[:slen_i]))
            #print('concatenating sil_mask shape: ', sil_mask.shape)
            #print('with curr_ph_seq_mask shape: ', curr_ph_seq_mask.shape)
            sil_mask = np.concatenate((sil_mask, 
                                       curr_ph_seq_mask.reshape((-1, 1))))
    return preds, gtruths, spks, sil_mask

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


def denorm_aco_preds_gtruth(preds, gtruths, spks, spk2acostats):
    # denorm based on spk_id stats
    for ii, (spk_i, aco_i, y_i) in enumerate(zip(spks, gtruths, 
                                                 preds)):
        aco_stats = spk2acostats[spk_i]
        aco_stats_aco = aco_stats['aco']
        #print('selected dur_stats {} for spk {}'.format(dur_stats, spk_i))
        aco_min = aco_stats_aco['min']
        aco_max = aco_stats_aco['max']
        preds[ii] = denorm_minmax(y_i, aco_min, aco_max)
        gtruths[ii] = denorm_minmax(aco_i, aco_min, aco_max)
    return preds, gtruths

def apply_pf(cc_pred, pf=1., n_feats=40):
    assert len(cc_pred.shape) == 2, cc_pred.shape
    pfs = [1.]
    for n in range(1, n_feats):
        pf_i = pf ** n
        pfs.append(pf_i)
        print('multiplying order {} by {}'.format(cc_pred[:, n], pf_i))
    cc_pred[:, :n_feats] = cc_pred[:, :n_feats] * pfs
    return cc_pred
