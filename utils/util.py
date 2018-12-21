import os
import torch
from .vocab import Vocab
import math


def load_word_vector(path):
    """
    loading word vector(this project employs GLOVE word vector), save GLOVE word, vector as file
    respectively
    :param path: GLOVE word vector path
    :return: glove vocab:vocab object, vector(torch tensor, of shape(words_num, word_dim))
    """
    base = os.path.splitext(os.path.basename(path))[0]
    glove_vocab_path = os.path.join('./data/glove/', base + '.vocab')
    glove_vector_path = os.path.join('./data/glove/', base + '.pth')
    # have loaded word vector
    if os.path.isfile(glove_vocab_path) and os.path.isfile(glove_vector_path):
        print('======> File found, loading memory !')
        vocab = Vocab(glove_vocab_path)
        vector = torch.load(glove_vector_path)
        return vocab, vector

    print('=====>Loading glove word vector<======')
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        contents = f.readline().rstrip('\n').split(' ')
        word_dim = len(contents[1:])
        count = 1
        for line in f:
            count += 1

    vocab = [None] * count
    vector = torch.zeros(count, word_dim, dtype=torch.float)
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            vocab[idx] = contents[0]
            vector[idx] = torch.tensor(list(map(float, contents[1:])), dtype=torch.float)
            idx += 1
    assert count == idx
    with open(glove_vocab_path, 'w') as f:
        for token in vocab:
            f.write(token + '\n')

    vocab = Vocab(glove_vocab_path)
    torch.save(vector, glove_vector_path)
    return vocab, vector


def build_vocab(filenames, vocabfile):
    """
    use train, dev, test file to build vocabulary
    :param filenames: files containing training, dev, test file.One sentence per line
    :param vocabfile: saved vocab path
    :return:
    """
    vocab = set()
    for filename in filenames:
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                words = line.strip().split(' ')
                vocab |= set(words)
    with open(vocabfile, 'w') as f:
        for word in sorted(vocab):
            f.write(word + '\n')


def map_target_to_prob(target, num_classes):
    """
    map true target into probability distribution
    :param target: a real num
    :param num_classes: the number of class
    :return: probability distribution of target (tensor with shape(1, num_classes))
    """
    prob = torch.zeros(1, num_classes, dtype=torch.float)
    ceil = int(math.ceil(target))
    floor = int(math.floor(target))
    if ceil == floor:
        prob[0, ceil - 1] = 1
    else:
        prob[0, floor - 1] = ceil - target
        prob[0, ceil - 1] = target - floor
    return prob




