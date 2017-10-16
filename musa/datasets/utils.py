import numpy as np
import pickle
import os
import re
import json
import pdb


class label_encoder(object):

    def __init__(self, codebooks_path=None, lab_data=None, 
                 force_gen=False):
        # if codebooks_path is specified, they don't have to be generated
        self.codebooks = {}
        self.lab_format = {1:'cate', 2:'cate', 3:'cate', 4:'cate', 5:'cate',
                           6:'real', 7:'real', 8:'bool', 9:'bool', 10:'real',
                           11:'bool', 12:'bool', 13:'real', 14:'real',
                           15:'real', 16:'real', 17:'real', 18:'real',
                           19:'real', 20:'real', 21:'real', 22:'real',
                           23:'real', 24:'real', 25:'real', 26:'cate',
                           27:'bool', 28:'bool', 29:'real', 30:'cate',
                           31:'real', 32:'cate', 33:'real', 34:'real',
                           35:'real', 36:'real', 37:'real', 38:'real',
                           39:'real', 40:'cate', 41:'real', 42:'real',
                           43:'real', 44:'real', 45:'real', 46:'real',
                           47:'real', 48:'cate', 49:'real', 50:'real',
                           51:'real', 52:'real', 53:'real'
                           }

        self.codebook_name = [ 'p1','p2','p3','p4','p5','p6','p7',
                               'a1','a2','a3','b1','b2','b3','b4','b5',
                               'b6','b7','b8','b9','b10','b11','b12',
                               'b13','b14','b15','b16','c1','c2',
                               'c3','d1','d2','e1','e2','e3','e4',
                               'e5','e6','e7','e8','f1', 'f2','g1',
                               'g2','h1','h2','h3','h4','h5','i1',
                               'i2','j1','j2','j3']

        if codebooks_path is None:
            raise ValueError('Please specify a codebooks path to load/save.')
        self.codebooks_path = codebooks_path
        if not os.path.exists(codebooks_path) or force_gen:
            if lab_data is None:
                raise ValueError('lab data is requierd in absence of codebooks'
                                 ' to make the codebooks!')
            self.codebooks = self.make_codebooks(lab_data)
            self.save_codebooks()
        else:
            self.load_codebooks()

    def save_codebooks(self):
        codebooks_path = self.codebooks_path
        with open(codebooks_path, 'wb') as cbooks_f:
            pickle.dump(self.codebooks, cbooks_f)
            print('Saved codebooks in ', codebooks_path)

    def load_codebooks(self):
        codebooks_path = self.codebooks_path
        with open(codebooks_path, 'rb') as cbooks_f:
            self.codebooks = pickle.load(cbooks_f)
            print('Loaded codebooks from ', codebooks_path)


    def make_codebooks(self, lab_data):
        """ Make codebooks out of lab training data

            # Arguments
                lab_data: list of label features extracted with parser
        """
        if len(lab_data) <= 0:
            raise ValueError('Must have lab data to make codebooks.')
        codebooks = {}
        num_lines = len(lab_data)
        print('Making codebooks in lab encoder with {} lab '
              'lines...'.format(num_lines))
        for line_i, lab_line in enumerate(lab_data, start=1):
            for lab_i, lab_el in enumerate(lab_line, start=1):
                try:
                    cbook = self.codebook_name[lab_i - 1] # indexed from 0
                    if self.lab_format[lab_i] == 'cate':
                        if cbook not in codebooks:
                            # init dictionary codebook
                            codebooks[cbook] = {'UNK':0}
                        # process categorical input adding to vocab if needed
                        if lab_el == 'UNKNOWN':
                            # ignore hard-coded UNKNOWN if appears
                            continue
                        if lab_el not in codebooks[cbook]:
                            # add new element in codebook for this idx
                            nel = np.max(list(codebooks[cbook].values())) + 1 
                            codebooks[cbook][lab_el] = nel
                    elif self.lab_format[lab_i] == 'real':
                        if cbook not in codebooks:
                            # init dictionary codebook
                            codebooks[cbook] = []
                        # accumulate value to compute statistics
                        codebooks[cbook].append(float(lab_el))
                        if line_i >= num_lines:
                            # save this final floats
                            #np.save('/tmp/{}_symbols'.format(cbook), codebooks[cbook])
                            # get unique symbols
                            codebooks[cbook] = list(set(codebooks[cbook]))
                            cbook_len = len(codebooks[cbook])
                            assert cbook_len > 0, cbook_len
                            # if last line, compute final stats
                            if len(codebooks[cbook]) > 1:
                                cbook_mean = np.mean(codebooks[cbook])
                                cbook_std = np.std(codebooks[cbook])
                            else:
                                cbook_mean = 0.
                                cbook_std = 1.
                            if cbook_std == 0:
                                # make no effect on znorm with std=1
                                cbook_std = 1
                            cbook_min = np.min(codebooks[cbook])
                            cbook_max = np.max(codebooks[cbook])
                            codebooks[cbook] = {'mean':cbook_mean,
                                                'std':cbook_std,
                                                'min':cbook_min,
                                                'max':cbook_max}
                    elif self.lab_format[lab_i] == 'bool':
                        # do nothing atm with bool entry (already normalized)
                        pass
                except KeyError:
                    # This exception means wrong formated lab line
                    raise
        return codebooks

    def __call__(self, lab_line, normalize='nonorm', sort_types=True,
                 verbose=False):
        return self.encode(lab_line, normalize=normalize, verbose=verbose,
                          sort_types=sort_types)

    def encode(self, lab_line, normalize='nonorm', sort_types=True, 
               verbose=False):
        """ Encode input label line with codebooks

            # Arguments
                lab_line: list of label elements
                normalize: Normalization option for real values.
                           options: znorm (mean=0, std=1),
                                    minmax (min=0, max=1),
                                    nonorm (do not).
                sort_types: sort lab elements by type (categories, real+bools)
                            instead of lab natural order interleaving types. In
                            case it is not sorted, one-hot codes are expanded,
                            not indexed.
                verbose: verbose or not the encoding process. Def False.

            # Return
                encoded lab list
        """
        encoded = []
        c_encoded = [] # categorical
        o_encoded = [] # others (real + bool)
        codebooks = self.codebooks
        lab_format = self.lab_format
        if verbose:
            print('-' * 50 + '\n' + '-' * 50)
        for lab_i, lab_el in enumerate(lab_line, start=1):
            try:
                cbook = self.codebook_name[lab_i - 1] # indexed from 0
                if lab_format[lab_i] == 'cate':
                    try:
                        cate_code = codebooks[cbook][lab_el]
                    except KeyError:
                        cate_code = codebooks[cbook]['UNK']
                    code_len = len(codebooks[cbook].keys())
                    if verbose:
                        print('Lab encoding {}: categorical element {} -> '
                              '{}'.format(cbook, lab_el, cate_code))
                    if sort_types:
                        c_encoded.append(cate_code)
                    else:
                        ohot = [0.] * code_len
                        ohot[cate_code] = 1.
                        encoded.extend(ohot)
                elif lab_format[lab_i] == 'real':
                    stats = codebooks[cbook]
                    lab_val = float(lab_el)
                    if normalize == 'nonorm':
                        real_code = lab_val
                    elif normalize == 'minmax':
                        den = stats['max'] - stats['min']
                        if den == 0.:
                            # min and max are the same
                            den = 1.
                        real_code = (lab_val - stats['min']) / den
                    elif normalize == 'znorm':
                        real_code = (lab_val - stats['mean']) / stats['std']
                    else:
                        raise ValueError('Unrecognized normalization '
                                         'method ', normalize)
                    if verbose:
                        print('Lab encoding {}: {} norm {} -> '
                              '{}'.format(cbook, normalize, lab_el, real_code))
                    if sort_types:
                        o_encoded.append(real_code)
                    else:
                        encoded.append(real_code)
                elif lab_format[lab_i] == 'bool':
                    # no norm, just pass value
                    if sort_types:
                        o_encoded.append(float(lab_el))
                    else:
                        encoded.append(float(lab_el))
            except KeyError:
                # This exception means wrong formated lab line
                raise
        if verbose:
            print('-' * 50 + '\n' + '-' * 50)
        if sort_types:
            return c_encoded + o_encoded
        else:
            return encoded

class label_parser(object):

    def __init__(self, ogmios_fmt = True):
        self.timestamp = '^\ (.*)\ (.*)\ (.*)'
        self.sec2 = 'A:(.*)\_(.*)\_(.*)/'
        self.sec4 = 'C:(.*)\+(.*)\+(.*)/'
        self.sec5 = 'D:(.*)\_(.*)/'
        self.sec7 = 'F:(.*)\_(.*)/'
        self.sec8 = 'G:(.*)\_(.*)/'
        self.sec11 = 'J:(.*)\+(.*)\-(.*)'
        if ogmios_fmt:
            print("LAB PARSER: Using Ogmios lab format")
            self.sec1 = '^(.*)\^(.*)\-(.*)\+(.*)\=(.*)\~(.*)\_(.*)/'
            self.sec3 = 'B:(.*)\-(.*)\-(.*)\~(.*)\-(.*)\&(.*)\-(.*)' \
                        '\#(.*)\-(.*)\$(.*)\-(.*)\!(.*)\-(.*)\;(.*)' \
                        '\-(.*)\|(.*)/'
            self.sec6 = 'E:(.*)\+(.*)\~(.*)\+(.*)\&(.*)\+(.*)\#(.*)\+(.*)/'
            self.sec9 = 'H:(.*)\=(.*)\~(.*)\=(.*)\|(.*)/'
            self.sec10 = 'I:(.*)\_(.*)/'
        else:
            print("LAB PARSER: Using Fst lab format")
            self.sec1 = '^(.*)\^(.*)\-(.*)\+(.*)\=(.*)\@(.*)\_(.*)/'
            self.sec3 = 'B:(.*)\-(.*)\-(.*)\@(.*)\-(.*)\&(.*)\-(.*)' \
                        '\#(.*)\-(.*)\$(.*)\-(.*)\!(.*)\-(.*)\;(.*)' \
                        '\-(.*)\|(.*)/'
            self.sec6 = 'E:(.*)\+(.*)\@(.*)\+(.*)\&(.*)\+(.*)\#(.*)\+(.*)/'
            self.sec9 = 'H:(.*)\=(.*)\@(.*)\=(.*)\|(.*)/'
            self.sec10 = 'I:(.*)\=(.*)/'

    def __call__(self, label_line, verbose=False):
        if type(label_line) == list:
            plab = []
            tss = []
            for line_i, lline in enumerate(label_line, start=1):
                if verbose:
                    print('Parsing labline {}/{}'.format(line_i,
                                                         len(label_line)))
                parsed = self.parse(lline)
                tss.append(parsed[0])
                plab.append(parsed[1])
            return tss, plab
        elif type(label_line) == str:
            return self.parse(label_line)
        else:
            raise TypeError('Either a list of lab lines or a lab line must '
                            'be passed to the parser')

    def parse(self, label_line):
        composed_regx = str(self.sec1 + self.sec2 + self.sec3 + self.sec4 + \
                            self.sec5 + self.sec6 + self.sec7 + self.sec8 + \
                            self.sec9 + self.sec10 + self.sec11)
        parsed_list = None
        #look for timestamp first and separate it (if exists) from ling. info
        tstamp_search = re.search(self.timestamp, label_line, 0)
        tstamp = None
        if tstamp_search:
            tstamp = [tstamp_search.group(1), tstamp_search.group(2)]
            # re-set the label_line as whatever came after the timestamp
            label_line = tstamp_search.group(3)
        # parse input label to get the individual elements
        lab_content = re.search(composed_regx,label_line, 0)
        if lab_content:
            parsed_list = []
            for n in range(0, len(lab_content.groups())):
                parsed_list.append(lab_content.group(n + 1))

        return tstamp, parsed_list

class querist(object):
    #TODO: get the size of answers IN CONSTRUCTOR, NOT ONCE ONE ANSWER
    #IS DONE
    def __init__(self,qstFile):
        self.readQuestions(qstFile)
        self.qstCodes = [
            "0;LL;{;^*",
            "1;L;*^;-*",
            "2;C;*-;+*",
            "3;R;*+;=*",
            "4;RR;*=;~*",
            "29;L-Word_GPOS;*/D:;_*",
            "31;C-Word_GPOS;*/E:;+*",
            "39;R-Word_GPOS;*/F:;_*"
        ]
        self.responsesLenghts = None


    def answer(self, labelList):
        response = None
        if(labelList is not None):
            #initialize sub types regexps
            qSubset = []
            response = ""
            self.responsesLenghts = []
            #question 1 to 8
            for n in range(len(self.qstCodes)):
                qSubset = []
                #keep track of the number of bits per response, so that we can
                #later know the length of the binary codes
                responseLength = 0
                code = self.qstCodes[n].split(';')
                if n<5:
                    #take into account that 1 to 5 are phonemes, so we must leave GPOS tags
                    #phoneme subset
                    qSubset.extend([s for s in self.questions if '\t\"'+code[1]+'-' in s and 'GPOS' not in s])
                else:
                    #POS subset
                    qSubset.extend([s for s in self.questions if '\t\"'+code[1] in s])
                for question in qSubset:
                    if str(code[2]+labelList[int(code[0])]+code[3]) in question:
                        response += '1\t'
                        responseLength += 1
                    else:
                        if (not n) and (str(','+labelList[int(code[0])]+code[3]) in question):
                            #special case of LL preceded by comma
                            response += '1\t'
                            responseLength += 1
                            continue
                        #negative question
                        response += '0\t'
                        responseLength += 1
                self.responsesLenghts.append(responseLength)
            response = response[:-1]

        return response

    def readQuestions(self, qstFile):
        qstF = open(qstFile, 'r')
        self.questions = qstF.readlines()
        qstF.close()


def to_lstm_bitstream(bitstream, questions=True, unk_phonemes=True):
    # takes a full label+questions bitstream and returns the lstm formatted
    # components, discarding any past elements p1,p2,d1,d2,g1,g2
    if type(bitstream) is list:
        bits = bitstream
    else:
        bits = bitstream.split('\t')
    # one hot size of regular phoneme (-1) for central phoneme
    if unk_phonemes:
        regular_ph_size = 35
    else:
        regular_ph_size = 34
    pos_size = 47
    lstm_bits = list(bits[regular_ph_size*2:regular_ph_size*5-1])
    lstm_bits.extend(bits[regular_ph_size*5-1:regular_ph_size*5-1+29])
    # here we jump the d1 and d2
    lstm_bits.extend(bits[regular_ph_size*5+29+pos_size:regular_ph_size*5+37+3*pos_size])
    # here we jump the g1 and g2
    lstm_bits.extend(bits[regular_ph_size*5+37+3*pos_size+2:regular_ph_size*5+37+3*pos_size+2+15])
    if questions:
        # get the question bits
        q_bits = bits[-189:]
        lstm_q_bits = q_bits[33*2:-24]
        # append the not-past questions
        lstm_q_bits.extend(q_bits[-16:])
        lstm_bits.extend(lstm_q_bits)
    return lstm_bits


def process_lab_line(lab_line, querist, l_encoder, l_parser,
                     special_phoneme=None):
    [time_stamp, label_list] = l_parser.parse(lab_line)
    sp_ph = None
    if special_phoneme:
        if special_phoneme == label_list[2]:
            sp_ph = True
        else:
            sp_ph = False
    answers = querist.answer(label_list)
    bitstream = l_encoder.encode(label_list)
    bitstream += '\t' + answers
    # just in case we want to check the label being processed
    return [time_stamp, bitstream, sp_ph]

def tstamps_to_dur(tstamps, flat_input=False):
    # convert the list of lists of tstamps [[beg_1, end_1], ...] to durs
    # in seconds
    # flat means there is one level of list, so list of lists of tstamps
    # no flat means there are two levels, list of lists of lists (3-D tensor)
    durs = []
    if flat_input:
        for t_i, tss in enumerate(tstamps):
            beg_t, end_t = map(float, tss)
            durs.append((end_t - beg_t) / 1e7)
    else:
        for seq in tstamps:
            #durs_t = []
            durs_t = tstamps_to_dur(seq, flat_input=True)
            #for t_i, tss in enumerate(seq):
            #    beg_t, end_t = map(float, tss)
            #    durs_t.append((end_t - beg_t) / 1e7)
            durs.append(durs_t)
    return durs

def trim_spk_samples(spk_samples, spk_phones, min_count, mulout):
    """ Trim each spk samples to min_count """
    # keep track of retained samples per spk
    spk_counts = {}
    if mulout:
        # multiple outputs
        trim_samples = {}
        trim_phones = {}
        assert isinstance(spk_samples, dict), type(spk_samples)
        for spk_name, spk_sample in spk_samples.items():
            print('spk_name: ', spk_name)
            print('spk len: ', len(spk_sample))
            print('min_count to apply: ', min_count)
            print('spk_sample type: ', type(spk_sample))
            for spk_sample_, spk_phone_ in zip(spk_sample,
                                               spk_phones[spk_name]):
                if spk_name not in spk_counts:
                    spk_counts[spk_name] = 0
                    trim_samples[spk_name] = []
                    trim_phones[spk_name] = []

                if spk_counts[spk_name] < (min_count - 1):
                    trim_samples[spk_name].append(spk_sample_)
                    trim_phones[spk_name].append(spk_phone_)
                    spk_counts[spk_name] += 1

            #spk_trim_sample, \
            #spk_trim_phones = trim_spk_samples(spk_sample,
            #                                   spk_phones[spk_name],
            #                                   min_count,
            #                                   False)
            #assert len(spk_trim_sample) == min_count, len(spk_trim_sample)
            #assert len(spk_trim_phones) == min_count, len(spk_trim_phones)
            #trim_samples[spk_name] = spk_trim_sample[:]
            #trim_phones[spk_name] = spk_trim_phones[:]
    else:
        # process one-by-one
        trim_samples = []
        trim_phones = []
        print('spk samples len: ', len(spk_samples))
        print('spk phones len: ', len(spk_phones))
        print('trimming to min count: ', min_count)
        for spk_sample, spk_phone in zip(spk_samples, spk_phones):
            #pdb.set_trace()
            spk_id = int(spk_sample[0][0])
            if spk_id not in spk_counts:
                spk_counts[spk_id] = 0
            #print('Spk_counts[{}] = {}'.format(spk_id,
            #                                   spk_counts[spk_id],
            #                                   min_count - 1))
            if spk_counts[spk_id] < (min_count - 1):
                trim_samples.append(spk_sample)
                trim_phones.append(spk_phone)
                spk_counts[spk_id] += 1
    return trim_samples, trim_phones

def statefulize_data(data, batch_size, seq_len):
    assert isinstance(data, dict), type(data)
    st_data = dict(data)
    for datak, datav in data.items():
        data_vals = datav['data']
        np_class = datav['np_class']
        data_arr = np_class(data_vals)
        #print('data_arr shape: ', data_arr.shape)
        #print('np type of {}: {}'.format(datak, type(data_arr)))
        data_arr = data_arr.reshape((batch_size, -1, data_arr.shape[-1]))
        #print('{}_arr shape: {}'.format(datak, data_arr.shape))
        data_arr = np.split(data_arr, data_arr.shape[1] // seq_len, axis=1)
        #print('Interleaved {}_arr[0] shape: {}'.format(datak, data_arr[0].shape))
        data_arr = np.concatenate(data_arr, axis=0)
        #print('Interleaved {}_arr:{}'.format(datak, data_arr.shape))
        st_data[datak]['st_data'] = data_arr
    return st_data

