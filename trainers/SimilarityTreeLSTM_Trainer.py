import torch
import os
import torch.utils.data as data
from tqdm import tqdm
from utils import util
import logging


class SimilarityTreeLSTM_Trainer(object):
    """
    Similarity Tree LSTM model trainer
    """
    def __init__(self, model, vocab, num_classes, criterion, optimizer,
                 train_dataset, dev_dataset, test_dataset, epoches,
                 metrics, args):
        """

        :param model: Similarity Tree LSTM model instance
        :param vocab: vocabulary
        :param num_classes: the number of class
        :param criterion: loss function
        :param optimizer: optimizer
        :param train_dataset: training dataset
        :param dev_dataset: dev dataset
        :param test_dataset: test dataset
        :param epoches: training epoch
        :param metrics: evaluate metrics
        :param args: arg
        """
        self.model = model
        self.vocab = vocab
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.epoches = epoches
        self.metrics = metrics
        self.args = args


    def train(self):
        """
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        best = -float('inf')
        for epoch in range(self.epoches):
            train_loss = self.train_epoch(epoch)
            train_loss, train_pred = self.test_epoch(self.train_dataset, epoch)
            dev_loss, dev_pred = self.test_epoch(self.dev_dataset, epoch)
            test_loss, test_pred = self.test_epoch(self.test_dataset, epoch)

            # pearson correlation
            train_pearson = self.metrics.pearson(train_pred, self.train_dataset.sims)
            dev_pearson = self.metrics.pearson(dev_pred, self.dev_dataset.sims)
            test_pearson = self.metrics.pearson(test_pred, self.test_dataset.sims)

            # mse
            train_mse = self.metrics.mse(train_pred, self.train_dataset.sims)
            dev_mse = self.metrics.mse(dev_pred, self.dev_dataset.sims)
            test_mse = self.metrics.mse(test_pred, self.test_dataset.sims)

            logger.info('==> Epoch {}, Training \tLoss: {}\tPearson: {}\tMSE: {}'.format(
                epoch, train_loss, train_pearson, train_mse
            ))
            logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(
                epoch, dev_loss, dev_pearson, dev_mse
            ))
            logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(
                epoch, test_loss, test_pearson, test_mse
            ))
            if best < test_pearson:
                best = test_pearson
                # save the best model
                checkpoint = {
                    'model' : self.model.state_dict(),
                    'optim' : self.optimizer,
                    'pearson' : test_pearson, 'mse' : test_mse,
                    'args' : self.args, 'epoch' : epoch
                }
                logger.debug('==> New optimum found, checkpointing everything now...')
                torch.save(checkpoint, '%s.pt' % os.path.join(self.args.save, self.args.expname))

    def train_epoch(self, epoch):
        """
        Traing model
        :param epoches: current epoch number
        :return: training_loss
        """
        # set the model in training mode
        self.model.train()
        self.optimizer.zero_grad()
        training_loss = 0.0
        data_size = len(self.train_dataset)
        indices = torch.randperm(data_size, dtype=torch.long)
        for idx in tqdm(range(data_size), desc='Training epoch ' + str(epoch +1) + ''):
            lsent, ltree, rsent, rtree, sim = self.train_dataset[indices[idx]]
            prob = self.model(lsent, ltree, rsent, rtree)
            target = util.map_target_to_prob(sim, self.num_classes)
            loss = self.criterion(prob, target)
            training_loss += loss.item()
            loss.backward()
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return training_loss / data_size


    def test_epoch(self, dataset, epoch):
        """
        testing after one epoch training
        :param dataset: data set
        :param epoch: current epoch number
        :return: total_loss, pred
        """
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            pred = torch.zeros(len(dataset), dtype=torch.float)
            r = torch.arange(1, self.num_classes + 1, dtype=torch.float)
            for idx in tqdm(range(len(dataset)), desc='Test epoch ' + str(epoch + 1) + ''):
                lsent, ltree, rsent, rtree, sim = dataset[idx]
                target = util.map_target_to_prob(sim, self.num_classes)
                prob = self.model(lsent, ltree, rsent, rtree)
                loss = self.criterion(prob, target)
                total_loss += loss.item()
                prob = prob.squeeze()
                score = torch.dot(r, torch.exp(prob))
                pred[idx] = score
        return total_loss / len(dataset), pred



