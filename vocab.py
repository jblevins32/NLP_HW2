from typing import List
from collections import Counter
from itertools import chain
import json
import torch

from loss import input_transpose


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3

        self.unk_id = self.word2id['<unk>']

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = input_transpose(word_ids, self['<pad>'])

        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)

        return sents_var

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry


class Vocab(object):
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
        '''
        In this function, you need to build the vocabulary for both source and target.
        You need to call VocabEntry.from_corpus to build the vocabulary for each language.
        :param src_sents: (List[List[str]]) list of source sentences
        :param tgt_sents: (List[List[str]]) list of target sentences
        :param vocab_size: (int) size of the vocabulary
        :param freq_cutoff: (int) frequency cutoff
        :return: (Vocabulary object) a Vocabulary object
        '''
        assert len(src_sents) == len(tgt_sents)

        src = None
        tgt = None

        # YOUR CODE STARTS HERE #

        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

        # YOUR CODE ENDS HERE #

        return Vocab(src, tgt)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))
