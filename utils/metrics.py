import copy
import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        """
        pearson metric
        :param predictions: tensor,of shape [num_classes]
        :param labels: tensor,of shape [num_classes]
        :return: pearson correlation
        """
        pred = copy.deepcopy(predictions)
        target = copy.deepcopy(labels)
        cov_pred = (pred - pred.mean()) / pred.std()
        cov_tar = (target - target.mean()) / target.std()
        return torch.mean(torch.mul(cov_pred, cov_tar))

    def mse(self, predictions, labels):
        """
        mse metric
        :param predictions: of shape [num_classes]
        :param labels: of shape [num_classes]
        :return: mse correlation
        """
        pred = copy.deepcopy(predictions)
        target = copy.deepcopy(labels)
        return torch.mean((pred - target) ** 2)
