

class Vocab(object):
    """
    vocab object
    """
    def __init__(self, filename=None, special_words=None, lower=False):
        """

        :param filename: vocab filepath
        :param special_word: special words, containing: EOS_WORD, BOS_WORD, UNK_WORD, PAD_WORD
        :param lower: whether lower
        """
        self.WordsToIdx = {}
        self.IdxToWords = {}
        self.lower = lower
        if special_words is not None:
            for special_word in special_words:
                self.add(special_word)

        if filename is not None:
            self.loadfile(filename)

        self.size = self.get_size()

    def get_size(self):
        return len(self.WordsToIdx)

    def loadfile(self, filename=None):
        """
        Read the vocab to build the vocab object
        :param filename: vocab file path
        :return:
        """
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            for word in f:
                word = word.strip()
                if word == '':
                    continue
                word = word.lower() if self.lower else word
                self.add(word)

    def add(self, word):
        """
        Add word into Vocab, i.e. add word into WordToIdx, IdxToWord
        :param word: word
        :return:
        """
        if word not in self.WordsToIdx:
            idx = len(self.IdxToWords)
            self.WordsToIdx[word] = idx
            self.IdxToWords[idx] = word

    def get_index(self, word, default=None):
        """
        Get the word index
        :param word: word
        :param default: if the word is not in the vocab, then return default
        :return: word index
        """
        if word in self.WordsToIdx:
            return self.WordsToIdx[word]
        else:
            return default

    def get_word(self, index, default=None):
        """
        Get the word according to index
        :param index: word index
        :param default: if the word is not in the vocab, and then return default
        :return: word
        """
        if index in self.IdxToWords:
            return self.IdxToWords[index]
        else:
            return default

    def convert_tokens_to_idx(self, tokens, unkWord, bosWord=None, eosWord=None):
        """
        convert tokens into index, if token is not found in the vocab, and then use unkWord to
        replace
        option: at the beginning and end of sentence, pad bosWord and eosWord respectivelly
        :param tokens: list contain token
        :param unkWord: if token is not found in the vocab, and the use unkWord to replace
        :param bosWord: begining word of sentence
        :param eosWord: ending word of sentence
        :return:list, containing index of tokens
        """
        indices = []
        unk_index = self.get_index(unkWord)

        if bosWord is not None:
            bos_index = self.get_index(bosWord)
            indices.append(bos_index)

        for token in tokens:
            token_index = self.get_index(token, unk_index)
            indices.append(token_index)

        if eosWord is not None:
            eos_index = self.get_index(eosWord)
            indices.append(eos_index)

        return indices

    def convert_idx_to_tokens(self, indices, unk_word):
        """
        Convert Indices into tokens, if index is not in the vocab, and then use unk_word to replace
        :param indices: list, containing indices
        :param unk_word: if index is not in the vocab, and then use unk_word to replace
        :return: list, containing words
        """
        words = []
        for index in indices:
            token = self.get_word(index, unk_word)
            words.append(token)

        return words









