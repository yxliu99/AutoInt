import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON1 = 1e-10
EPSILON2 = 1e-3  # or 1e-2


class Binarize(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > 0, torch.ones_like(X), torch.zeros_like(X))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Enforced(torch.autograd.Function):
    """Deterministic binarization."""
    @staticmethod
    def forward(ctx, X):
        y = torch.where(X > EPSILON2, X, torch.full_like(X, EPSILON2))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class BinarizeLayer(nn.Module):
    def __init__(self, input_dim, i_min, i_max, t1, t2, interval, interval2, use_not=False):
        super(BinarizeLayer, self).__init__()
        self.interval_number1 = len(interval[0])
        if interval2:
            self.interval_number2 = len(interval2[0])
        else:
            self.interval_number2 = 0
        self.input_dim = input_dim
        self.disc_num = input_dim[0]
        self.con_num = input_dim[1]
        self.use_not = use_not
        if self.use_not:
            self.disc_num *= 2
        self.output_dim = self.disc_num + (self.interval_number1 + self.interval_number2) * self.con_num
        self.i_min = i_min
        self.i_max = i_max
        self.t1 = t1
        self.t2 = t2
        self.layer_type = 'binarization'
        self.dim2id = {i: i for i in range(self.output_dim)}

        self.interval = nn.Parameter(torch.Tensor(interval), requires_grad=True)
        if interval2:
            self.interval2 = nn.Parameter(torch.Tensor(interval2), requires_grad=True)
        else:
            self.interval2 = None

    def forward(self, x):
        if self.con_num > 0:
            x_disc, x = x[:, 0: self.input_dim[0]], x[:, self.input_dim[0]:]
            if self.use_not:
                x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
            interval_pos = Enforced.apply(self.interval)

            kmeans_loss = None
            for fea in range(self.con_num):
                one_feature_x = x[:, fea]
                one_feature_interval = interval_pos[fea]
                start = self.i_min[fea]
                for val in range(self.interval_number1):
                    if val == 0:
                        start += one_feature_interval[val]
                        Distance = (one_feature_x - start) ** 2
                        Distance = Distance.unsqueeze(-1)
                    else:
                        start += one_feature_interval[val]
                        dis = (one_feature_x - start) ** 2
                        dis = dis.unsqueeze(-1)
                        Distance = torch.cat((Distance, dis), dim=1)

                Dis_min = torch.min(Distance, dim=1)[0].unsqueeze(-1)
                Dis_exp = torch.exp(-self.t1 * (Distance - Dis_min))
                Dis_exp_sum = torch.sum(Dis_exp, dim=1).unsqueeze(-1)
                Dis_softmax = Dis_exp / Dis_exp_sum
                loss = Distance * Dis_softmax
                if fea == 0:
                    kmeans_loss = torch.mean(torch.sum(loss, dim=1), dim=0)
                else:
                    kmeans_loss += torch.mean(torch.sum(loss, dim=1), dim=0)

                X_exp = torch.exp(-self.t2 * (Distance - Dis_min))
                X_exp_sum = torch.sum(X_exp, dim=1).unsqueeze(-1)
                X_softmax = X_exp / X_exp_sum
                X_argmax = torch.argmax(X_softmax, dim=1)
                X_b = F.one_hot(X_argmax, num_classes=self.interval_number1)
                out = X_b.detach() + X_softmax - X_softmax.detach()
                if fea == 0:
                    total_out = out
                else:
                    total_out = torch.cat((total_out, out), dim=1)

            if self.interval_number2:
                kmeans_loss2 = None
                interval_pos2 = Enforced.apply(self.interval2)
                for fea in range(self.con_num):
                    one_feature_x = x[:, fea]  # shape:(batch_size)
                    one_feature_interval = interval_pos2[fea]
                    start = self.i_min[fea]
                    for val in range(self.interval_number2):
                        if val == 0:
                            start += one_feature_interval[val]
                            Distance = (one_feature_x - start) ** 2
                            Distance = Distance.unsqueeze(-1)
                        else:
                            start += one_feature_interval[val]
                            dis = (one_feature_x - start) ** 2
                            dis = dis.unsqueeze(-1)
                            Distance = torch.cat((Distance, dis), dim=1)

                    Dis_min = torch.min(Distance, dim=1)[0].unsqueeze(-1)
                    Dis_exp = torch.exp(-self.t1 * (Distance - Dis_min))
                    Dis_exp_sum = torch.sum(Dis_exp, dim=1).unsqueeze(-1)
                    Dis_softmax = Dis_exp / Dis_exp_sum
                    loss = Distance * Dis_softmax
                    if fea == 0:
                        kmeans_loss2 = torch.mean(torch.sum(loss, dim=1), dim=0)
                    else:
                        kmeans_loss2 += torch.mean(torch.sum(loss, dim=1), dim=0)

                    X_exp = torch.exp(-self.t2 * (Distance - Dis_min))
                    X_exp_sum = torch.sum(X_exp, dim=1).unsqueeze(-1)
                    X_softmax = X_exp / X_exp_sum
                    X_argmax = torch.argmax(X_softmax, dim=1)
                    X_b = F.one_hot(X_argmax, num_classes=self.interval_number2)
                    out = X_b.detach() + X_softmax - X_softmax.detach()
                    if fea == 0:
                        total_out_2 = out
                    else:
                        total_out_2 = torch.cat((total_out_2, out), dim=1)

                total_out = torch.cat((total_out, total_out_2), dim=1)
                kmeans_loss += kmeans_loss2

            return torch.cat((x_disc, total_out), dim=1), kmeans_loss
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x, None

    def binarized_forward(self, x):
        with torch.no_grad():
            return self.forward(x)

    def clip(self):
        pass

    def get_bound_name(self, feature_name, mean=None, std=None):
        bound_name = []
        for i in range(self.input_dim[0]):
            bound_name.append(feature_name[i])
        if self.use_not:
            for i in range(self.input_dim[0]):
                bound_name.append('~' + feature_name[i])
        if self.input_dim[1] > 0:

            interval = torch.where(self.interval > EPSILON2, self.interval,
                                   torch.full_like(self.interval, EPSILON2))
            interval = interval.detach().cpu().numpy()
            for i, ii in enumerate(interval):
                fi_name = feature_name[self.input_dim[0] + i]
                mini = self.i_min[i]
                maxi = self.i_max[i]
                interval_list = []
                for j in range(len(ii)):
                    if j == 0:
                        cl = mini
                        cr = cl + ii[j] + 1 / 2 * ii[j + 1]
                        interval_list.append((cl, cr))
                    elif j == len(ii) - 1:
                        cl = cr
                        cr = maxi
                        interval_list.append((cl, cr))
                    else:
                        cl = cr
                        cr += 1/2 * (ii[j] + ii[j + 1])
                        interval_list.append((cl, cr))
                for cl, cr in interval_list:
                    if mean is not None and std is not None:
                        cl = cl * std[fi_name] + mean[fi_name]
                        cr = cr * std[fi_name] + mean[fi_name]
                    bound_name.append('{:.3f} < {} < {:.3f}'.format(cl, fi_name, cr))
            if self.interval_number2:
                interval2 = torch.where(self.interval2 > EPSILON2, self.interval2,
                                       torch.full_like(self.interval2, EPSILON2))
                interval2 = interval2.detach().cpu().numpy()
                for i, ii in enumerate(interval2):
                    fi_name = feature_name[self.input_dim[0] + i]
                    mini = self.i_min[i]
                    maxi = self.i_max[i]
                    interval_list2 = []
                    for j in range(len(ii)):
                        if j == 0:
                            cl = mini
                            cr = cl + ii[j] + 1 / 2 * ii[j + 1]
                            interval_list2.append((cl, cr))
                        elif j == len(ii) - 1:
                            cl = cr
                            cr = maxi
                            interval_list2.append((cl, cr))
                        else:
                            cl = cr
                            cr += 1 / 2 * (ii[j] + ii[j + 1])
                            interval_list2.append((cl, cr))
                    for cl, cr in interval_list2:
                        if mean is not None and std is not None:
                            cl = cl * std[fi_name] + mean[fi_name]
                            cr = cr * std[fi_name] + mean[fi_name]
                        bound_name.append('{:.3f} < {} < {:.3f}'.format(cl, fi_name, cr))
        return bound_name


class Product(torch.autograd.Function):
    """Tensor product function."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON1))
        return grad_input


class EstimatedProduct(torch.autograd.Function):
    """Tensor product function with a estimated derivative."""
    @staticmethod
    def forward(ctx, X):
        y = (-1. / (-1. + torch.sum(torch.log(X), dim=1)))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y, = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * ((-1. / (-1. + torch.log(y.unsqueeze(1) ** 2))) / (X + EPSILON1))
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, n, input_dim):
        super(LRLayer, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.output_dim = self.n
        self.layer_type = 'linear'

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.fc1(x)

    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(ConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'conjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return self.Product.apply(1 - (1 - x).unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(DisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = self.n
        self.layer_type = 'disjunction'

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.n, self.input_dim))
        self.Product = EstimatedProduct if estimated_grad else Product
        self.node_activation_cnt = None

    def forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - self.Product.apply(1 - x.unsqueeze(-1) * self.W.t())

    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarize.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb.t(), dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


def extract_rules(prev_layer, skip_connect_layer, layer, layer_type, pos_shift=0):
    dim2id = defaultdict(lambda: -1)
    rules = {}
    tmp = 0
    rule_list = []
    if layer_type == 'conjunction':
        Wb = (layer.W > THRESHOLD).type(torch.int).detach().cpu().numpy()
    else:
        Wb = (layer.W > THRESHOLD).type(torch.int).detach().cpu().numpy()

    if skip_connect_layer is not None:
        shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
        prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
    else:
        merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

    for ri, row in enumerate(Wb):
        if layer.node_activation_cnt[ri + pos_shift] == 0 or layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot:
            dim2id[ri + pos_shift] = -1
            continue
        rule = {}
        for i, w in enumerate(row):
            if w > 0 and merged_dim2id[i][1] != -1:
                rule[merged_dim2id[i]] = 1
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list


class UnionLayer(nn.Module):
    """The union layer is used to learn the rule-based representation."""

    def __init__(self, n, input_dim, use_not=False, estimated_grad=False):
        super(UnionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim
        self.output_dim = self.n * 2
        self.layer_type = 'union'
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        self.con_layer = ConjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad)
        self.dis_layer = DisjunctionLayer(self.n, self.input_dim, use_not=use_not, estimated_grad=estimated_grad)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat([self.con_layer.binarized_forward(x),
                          self.dis_layer.binarized_forward(x)], dim=1)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = self.node_activation_cnt

        con_dim2id, con_rule_list = extract_rules(prev_layer, skip_connect_layer, self.con_layer, 'conjunction')
        dis_dim2id, dis_rule_list = extract_rules(prev_layer, skip_connect_layer, self.dis_layer, 'disjunction', self.con_layer.W.shape[0])

        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})
        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

        return dim2id, rule_list

    def get_rule_description(self, prev_rule_name, wrap=False):
        self.rule_name = []
        for rl, op in zip(self.rule_list, ('&', '|')):
            for rule in rl:
                name = ''
                for i, ri in enumerate(rule):
                    op_str = ' {} '.format(op) if i != 0 else ''
                    var_str = ('({})' if wrap else '{}').format(prev_rule_name[2 + ri[0]][ri[1]])
                    name += op_str + var_str
                self.rule_name.append(name)
