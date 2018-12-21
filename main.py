import torch
import os
import logging
from configs import config
from utils import util
from trainers import SimilarityTreeLSTM_Trainer
from utils.vocab import Vocab
from utils import Constants
from data_loader.SICKDataset import SICKDataset
import torch.utils.data.dataloader as dataloader
from models.SimilarityTreeLSTM import SimilarityTreeLSTM
import torch.nn as nn
from trainers.SimilarityTreeLSTM_Trainer import SimilarityTreeLSTM_Trainer
from utils.metrics import Metrics
import random
import torch.optim as optim



def main():
    """
    Main Function
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    global args
    args = config.parse_args()
    torch.manual_seed(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # console logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # build the vocab
    train_dir = os.path.join(args.data, 'train')
    dev_dir = os.path.join(args.data, 'dev')
    test_dir = os.path.join(args.data, 'test')

    vocab_file = os.path.join(args.data, 'sick.vocab')
    if not os.path.isfile(vocab_file):
        tokens_a_file = [os.path.join(dirname, 'a.toks') for dirname in [train_dir, dev_dir, test_dir]]
        tokens_b_file = [os.path.join(dirname, 'b.toks') for dirname in [train_dir, dev_dir, test_dir]]
        token_files = tokens_a_file + tokens_b_file
        util.build_vocab(token_files, vocab_file)

    vocab = Vocab(filename=vocab_file, special_words=[Constants.PAD_WORD, Constants.UNK_WORD,
                                                      Constants.BOS_WORD, Constants.EOS_WORD])

    logger.debug('==> SICK vocabulary size : %d' % vocab.size)

    # load the train dataset
    train_file = os.path.join(args.data, 'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(vocab, os.path.join(train_dir, 'a.toks'),
                                    os.path.join(train_dir, 'b.toks'), os.path.join(train_dir, 'sim.txt'),
                                    os.path.join(train_dir, 'a.parents'), os.path.join(train_dir, 'b.parents'))
        torch.save(train_dataset, train_file)

    logger.debug('==> Size of train data    : %d' % len(train_dataset))

    # load the dev data
    dev_file = os.path.join(args.data, 'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(vocab, os.path.join(dev_dir, 'a.toks'),
                                  os.path.join(dev_dir, 'b.toks'), os.path.join(dev_dir, 'sim.txt'),
                                  os.path.join(dev_dir, 'a.parents'), os.path.join(dev_dir, 'b.parents'))
        torch.save(dev_dataset, dev_file)

    logger.debug('==> Size of dev data      : %d' % len(dev_dataset))

    # load the test data
    test_file = os.path.join(args.data, 'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(vocab, os.path.join(test_dir, 'a.toks'),
                                   os.path.join(test_dir, 'b.toks'), os.path.join(test_dir, 'sim.txt'),
                                   os.path.join(test_dir, 'a.parents'), os.path.join(test_dir, 'b.parents'))
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data     : %d ' % len(test_dataset))


    # define the model
    model = SimilarityTreeLSTM(vocab, args.input_dim, args.mem_dim, args.hidden_dim,
                               args.num_classes, args.freeze_embed, args.sparse)

    # load the GLOVE word embedding
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    # special word vector defines zero
    sick_emb = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(sick_emb):
        emb = torch.load(sick_emb)
    else:
        glove_path = os.path.join(args.glove, 'glove.840B.300d.txt')
        glove_vocab, glove_vector = util.load_word_vector(glove_path)
        emb = torch.zeros(vocab.size, args.input_dim, dtype=torch.float)
        emb.normal_(0, 0.05)
        for idx in [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS]:
            emb[idx].zero_()
        for word in vocab.WordsToIdx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_vector[glove_vocab.get_index(word)]
        torch.save(emb, sick_emb)

    model.emb.weight.data.copy_(emb)

    # define loss function
    criterion = nn.KLDivLoss()

    # define evaluate metrics
    metrics = Metrics(args.num_classes)

    # define optimizer
    optimizer = optim.Adagrad(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                              weight_decay=args.wd)

    # define trainer
    sim_trainer = SimilarityTreeLSTM_Trainer(model, vocab, args.num_classes, criterion,
                                             optimizer, train_dataset,
                                             dev_dataset, test_dataset, args.epochs, metrics, args)

    # train model
    sim_trainer.train()


if __name__ == '__main__':
    main()

