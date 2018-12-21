import torch
import os
import torch.utils.data as data
from utils import Constants
from copy import deepcopy
from modules.tree import Tree


class SICKDataset(data.Dataset):
    """
    SICK dataset
    """
    def __init__(self, vocab, l_token_file, r_token_file, sim_file,
                 l_parent_file, r_parent_file):
        """
        create a SICK dataset instance
        :param vocab: vocabulary, vocab object
        :param l_token_file: left tokens file path
        :param r_token_file: right tokens file path
        :param sim_file: similarity file path
        :param l_parent_file: node parent in left sentences file path
        :param r_parent_file: node parent in right sentences file path
        """
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.lsentences = self.read_sentences(l_token_file)
        self.rsentences = self.read_sentences(r_token_file)
        self.sims = self.read_sim(sim_file)
        self.ltrees = self.read_trees(l_parent_file)
        self.rtrees = self.read_trees(r_parent_file)

    def __len__(self):
        return len(self.lsentences)

    def __getitem__(self, item):
        lsent = deepcopy(self.lsentences[item])
        ltree = deepcopy(self.ltrees[item])
        rsent = deepcopy(self.rsentences[item])
        rtree = deepcopy(self.rtrees[item])
        sim = deepcopy(self.sims[item])
        return lsent, ltree, rsent, rtree, sim

    def read_sentences(self, filename):
        """
        Read all sentences from file
        :param filename: one sentence per line in file
        :return: sentences: list, the element is sentence that encode torch tensor
        """
        sentences = list()
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                sentence = self.read_sentence(tokens)
                sentences.append(sentence)
        return sentences

    def read_sentence(self, tokens):
        """
        transform sentence into word indices
        :param tokens: list, containing words in sentence
        :return: words, torch tensor, of shape [num_words]
        """
        token_indices = self.vocab.convert_tokens_to_idx(tokens, Constants.UNK_WORD)
        words = torch.tensor(token_indices, dtype=torch.long)
        return words

    def read_trees(self, parent_file):
        """
        Build the dependency tree of all sentences
        :param parent_file: parent node file
        :return: list, the element is tree object of one sentence
        """
        trees = list()
        with open(parent_file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                parents = line.rstrip('\n').split()
                tree = self.read_tree(parents)
                trees.append(tree)
        return trees

    def read_tree(self, parents):
        """
        Construct the dependency tree of one sentence
        :param parents: list, the element is parent node of current node
        :return: root, tree object
        """
        root = None
        trees = dict()
        parents = list(map(lambda x: int(x), parents))
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent

        return root

    def read_sim(self, sim_file):
        """
        Read the similarity of every pair of sentences
        :param sim_file: similarity file
        :return: sims, torch tensor, of shape [num_sentences]
        """
        with open(sim_file, 'r', encoding='utf8', errors='ignore') as f:
            sims = [float(line.strip()) for line in f]
            sims = torch.tensor(sims, dtype=torch.float)
        return sims












