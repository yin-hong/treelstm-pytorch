import torch
import torch.nn as nn
import torch.nn.functional as F


class ChildSumTreeLSTM(nn.Module):
    """
    Child-Sum Tree-LSTMs
    """
    def __init__(self, input_dim, memory_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.ioux = nn.Linear(self.input_dim, 3 * self.memory_dim)
        self.iouh = nn.Linear(self.memory_dim, 3 * self.memory_dim)
        self.fx = nn.Linear(self.input_dim, self.memory_dim)
        self.fh = nn.Linear(self.memory_dim, self.memory_dim)

    def node_forward(self, child_h, child_c, input):
        """
        forward in one node of the tree
        :param child_h: children hidden state of node, torch tensor, of shape (num_child, mem_dim)
        :param child_c: child cell state of node, torch tensor, of shape (num_child, mem_dim)
        :param input: word embedding in this node, torch tensor, of shape (1, input_dim)
        :return: c, h: cell state, hidden state of shape(1, mem_dim)
        """
        h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(input) + self.iouh(h_sum)
        i, o, u = torch.split(iou, self.memory_dim, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        f = F.sigmoid(
            self.fh(child_h) + self.fx(input).repeat(child_h.size(0), 1)
        )

        c = torch.mul(i, u) + torch.sum(torch.mul(f, child_c), dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, inputs, sent_tree):
        """
        Child-Sum Tree-LSTMs model forward
        :param inputs: sentence, torch tensor, of shape (word_num, input_dim)
        :param sent_tree: sent_tree, tree object which parent is root node
        :return: c, h : this model cell state, hidden state
        """
        for i in range(len(sent_tree.children)):
            self.forward(inputs, sent_tree.children[i])
        # It is leaf node, h and c is initialized 0
        if len(sent_tree.children) == 0:
           child_h = torch.zeros(1, self.memory_dim, dtype=torch.float).requires_grad_()
           child_c = torch.zeros(1, self.memory_dim, dtype=torch.float).requires_grad_()
        else:
            child_h, child_c = zip(*list(map(lambda x: (x.hidden_state, x.cell_state),
                                             sent_tree.children)))
            child_h, child_c = torch.cat(child_h, dim=0), torch.cat(child_c, dim=0)
        sent_tree.cell_state, sent_tree.hidden_state = self.node_forward(child_h, child_c,
                                                                         inputs[sent_tree.idx])
        return sent_tree.cell_state, sent_tree.hidden_state







