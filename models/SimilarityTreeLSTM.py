import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ChildSumTreeLSTM import ChildSumTreeLSTM
from modules.Similarity import Similarity
from utils import Constants

class SimilarityTreeLSTM(nn.Module):
    """
    Semantic Relatedness of Sentences Pairs Model
    """
    def __init__(self, vocab, input_dim, memory_dim, hidden_dim,
                 num_classes, freeze, sparse):
        """

        :param vocab: vocabulary, vocab object
        :param input_dim: word dimension
        :param memory_dim: tree node hidden state dimension
        :param hidden_dim: similarity model hidden dimension
        :param num_classes: the number of class
        :param freeze: whether update the word embedding
        :param sparse: word embedding sparse
        """
        super(SimilarityTreeLSTM, self).__init__()
        self.vocab = vocab
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.childsumtreeLstm = ChildSumTreeLSTM(self.input_dim, self.memory_dim)
        self.similarity = Similarity(self.memory_dim, self.hidden_dim, self.num_classes)
        self.emb = nn.Embedding(self.vocab.get_size(), input_dim,
                                padding_idx=Constants.PAD, sparse=sparse)
        if freeze:
            self.emb.weight.requires_grad = False

    def forward(self, left_sentence, left_tree, right_sentence, right_tree):
        """

        :param left_sentence: torch tensor, of shape [words_num]
        :param left_tree: tree object
        :param right_sentence: torch tensor, of shape [words_num]
        :param right_tree: tree object
        :return: prob: similarity probability
        """
        left_sent = self.emb(left_sentence)
        right_sent = self.emb(right_sentence)
        left_cell_state, left_hidden_state = self.childsumtreeLstm(left_sent, left_tree)
        right_cell_state, right_hidden_state = self.childsumtreeLstm(right_sent, right_tree)
        prob = self.similarity(left_cell_state, right_cell_state)
        return prob






