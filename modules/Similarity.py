import torch.nn as nn
import torch
import torch.nn.functional as F

class Similarity(nn.Module):
    """
    Similarity Model
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        """

        :param input_dim: input dimension
        :param hidden_dim: hidden dimension
        :param num_classes: the number of class
        """
        super(Similarity, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.h_s_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.h_s_2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.h_p = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, l_h, r_h):
        """
        forward function of the model
        :param l_h: the representation of left sentence
        :param r_h: the representation of right sentence
        :return: prob of class
        """
        h_mul = torch.mul(l_h, r_h)
        h_add = torch.abs(torch.add(l_h, -r_h))
        h_s = F.sigmoid(self.h_s_1(h_mul) + self.h_s_2(h_add))
        prob = F.log_softmax(self.h_p(h_s), dim=1)
        return prob



