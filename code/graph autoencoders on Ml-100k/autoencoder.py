import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import torch.nn.init as init
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import math

class InputDropout(nn.Module):
    def __init__(self, keep_prob):
        super(InputDropout, self).__init__()
        self.p = keep_prob

    def forward(self, inputs):
        x = inputs.clone()
        if self.training:
            random_tensor = self.p + torch.rand((inputs.size(0),))
            dropout_mask = torch.floor(random_tensor).bool()
            x[~dropout_mask] = 0.
            return x / self.p
        else:
            return x


class StackGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support,
                 dropout=0.,
                 use_bias=False, activation=lambda x:x):
        #use_bias = False, activation = lambda x: 1. / (1 + torch.exp(-x))):
        """对得到的每类评分使用级联的方式进行聚合

        Args:
        ----
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 评分的类别数，比如1~5分，值为5
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(StackGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.dropout = dropout
        self.use_bias = use_bias
        self.activation = activation
        #以下是添加的
        #self.scale =tf.placeholder(tf.float32)
        #self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        #self.x =tf.placeholder(tf.float32, [None, self.input_dim])
        # 将输入x加上噪声
        #self.x_noise = self.x + scale * tf.random_normal((input_dim,))

        # 以上是添加的
        assert output_dim % num_support == 0
                                                 # self.x_noise
        self.weight = nn.Parameter(torch.Tensor( input_dim , output_dim))
        if self.use_bias:
            self.bias_user = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_item = nn.Parameter(torch.Tensor(output_dim, ))
        self.dropout = InputDropout(1 - dropout)
        self.reset_parameters()


    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias_user)
            init.zeros_(self.bias_item)

    def forward(self, user_supports, item_supports, user_inputs, item_inputs,scale=0.1):
        """StackGCNEncoder计算逻辑

        Args:
            user_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的用户与商品邻接矩阵
            item_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的商品与用户邻接矩阵
            user_inputs (torch.Tensor): 用户特征的输入
            item_inputs (torch.Tensor): 商品特征的输入

        Returns:
            [torch.Tensor]: 用户的隐层特征
            [torch.Tensor]: 商品的隐层特征

        """
        #self.scale = tf.placeholder(tf.float32)
        assert len(user_supports) == len(item_supports) == self.num_support

        print("user", user_inputs)
        #插入的代码
        #x = tf.placeholder(tf.float32, [None, user_inputs])
       # y = tf.placeholder(tf.float32, [None,item_inputs])
       #x_noise = x + scale * tf.random_normal((user_inputs,))
        #y_noise = x + scale * tf.random_normal((item_inputs,))
       #

        #user_inputs = self.dropout(user_inputs)
        #print(user_inputs.shape )
        #item_inputs = self.dropout(item_inputs)
        #print(item_inputs)
        #user_inputs = self.dropout(x_noise )
        #item_inputs = self.dropout(y_noise)
       ################################
        ###########################################
        ####################################
       # randomlist = []
        #arraylen = len(user_inputs)
        # print(user_inputs)
        # print(arraylen)
        # print(len(item_inputs))
       # while len(randomlist) < (arraylen / 23):
            #random_int = np.random.randint(0, arraylen)
          #  if random_int not in randomlist:
                #randomlist.append(random_int)
        #for position in randomlist:
            #user_inputs[position] = 0  # 23%
        #user=user_inputs
        #return inputs

        #randomlist1 = []
        #arraylen1 = len(item_inputs)
        # print(arraylen)
        # print(len(item_inputs))
        #while len(randomlist1) < (arraylen / 29):
           # random_int1 = np.random.randint(0, arraylen1)
            #if random_int1 not in randomlist1:
               # randomlist1.append(random_int1)
        #for position1 in randomlist:
            #item_inputs[position1] = 0  # 29%
       # item=item_inputs
        #return item

        ###############################################
        #n = len(user_inputs)
        #random_idx = np.random.permutation(n)
        #frac = 10
        #zero_idx = random_idx[:round(n * frac / 100)]
        #user_inputs[zero_idx] = 0

        #m = len(item_inputs)
        #random_idx1 = np.random.permutation(m)
        #frac1 = 10
        #zero_idx1 = random_idx[:round(m * frac1 / 100)]
        #item_inputs[zero_idx1] = 0
        #########################################
      ##################################################
        user_hidden = []
        item_hidden = []
        weights = torch.split(self.weight, self.output_dim//self.num_support, dim=1)
        for i in range(self.num_support):
            tmp_u = torch.matmul(user_inputs, weights[i])
            tmp_v = torch.matmul(item_inputs, weights[i])
            #tmp_u = torch.matmul(user, weights[i])
            #tmp_v = torch.matmul(item, weights[i])
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u)
            user_hidden.append(tmp_user_hidden)
            item_hidden.append(tmp_item_hidden)

        user_hidden = torch.cat(user_hidden, dim=1)
        item_hidden = torch.cat(item_hidden, dim=1)

        user_outputs = self.activation(user_hidden)
        item_outputs = self.activation(item_hidden)

        if self.use_bias:
            user_outputs += self.bias_user
            item_outputs += self.bias_item

        return user_outputs, item_outputs


class SumGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support,
                 dropout=0.,
                 use_bias=False,activation=lambda x:torch.log(1+torch.exp(x))):
        """对得到的每类评分使用求和的方式进行聚合

        Args:
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            num_support (int): 评分的类别数，比如1~5分，值为5
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
        """
        super(SumGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.use_bias = use_bias
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(
            input_dim, output_dim * num_support))
        if self.use_bias:
            self.bias_user = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_item = nn.Parameter(torch.Tensor(output_dim, ))
        self.dropout = InputDropout(1 - dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias_user)
            init.zeros_(self.bias_item)

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):
        """SumGCNEncoder计算逻辑

        Args:
            user_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的用户与商品邻接矩阵
            item_supports (list of torch.sparse.FloatTensor):
                归一化后每个评分等级对应的商品与用户邻接矩阵
            user_inputs (torch.Tensor): 用户特征的输入
            item_inputs (torch.Tensor): 商品特征的输入

        Returns:
            [torch.Tensor]: 用户的隐层特征
            [torch.Tensor]: 商品的隐层特征
        """
        assert len(user_supports) == len(item_supports) == self.num_support
        user_inputs = self.dropout(user_inputs)
        item_inputs = self.dropout(item_inputs)

        user_hidden = []
        item_hidden = []
        weights = torch.split(self.weight, self.output_dim, dim=1)
        for i in range(self.num_support):
            w = sum(weights[:(i + 1)])
            tmp_u = torch.matmul(user_inputs, w)
            tmp_v = torch.matmul(item_inputs, w)
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u)
            user_hidden.append(tmp_user_hidden)
            item_hidden.append(tmp_item_hidden)

        user_hidden, item_hidden = sum(user_hidden), sum(item_hidden)
        user_outputs = self.activation(user_hidden)
        item_outputs = self.activation(item_hidden)

        if self.use_bias:
            user_outputs += self.bias_user
            item_outputs += self.bias_item

        return user_outputs, item_outputs


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.,
                 use_bias=False,activation=lambda x:x,
                 share_weights=False):
        """非线性变换层

        Args:
        ----
            input_dim (int): 输入的特征维度
            output_dim (int): 输出的特征维度，需要output_dim % num_support = 0
            use_bias (bool, optional): 是否使用偏置. Defaults to False.
            activation (optional): 激活函数. Defaults to F.relu.
            share_weights (bool, optional): 用户和商品是否共享变换权值. Defaults to False.
        
        """
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = activation

        self.share_weights = share_weights
        if not share_weights:
            self.weights_u = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.weights_v = nn.Parameter(torch.Tensor(input_dim, output_dim))
            if use_bias:
                self.user_bias = nn.Parameter(torch.Tensor(output_dim))
                self.item_bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.weights_u = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.weights_v = self.weights_u
            if use_bias:
                self.user_bias = nn.Parameter(torch.Tensor(output_dim))
                self.item_bias = self.user_bias
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if not self.share_weights:
            init.xavier_uniform_(self.weights_u)
            init.xavier_uniform_(self.weights_v)
            if self.use_bias:
                init.normal_(self.user_bias, std=0.5)
                init.normal_(self.item_bias, std=0.5)
        else:
            init.xavier_uniform_(self.weights_u)
            if self.use_bias:
                init.normal_(self.user_bias, std=0.5)

    def forward(self, user_inputs, item_inputs):
        """前向传播
        
        Args:
            user_inputs (torch.Tensor): 输入的用户特征
            item_inputs (torch.Tensor): 输入的商品特征
        
        Returns:
            [torch.Tensor]: 输出的用户特征
            [torch.Tensor]: 输出的商品特征
        """
        x_u = self.dropout(user_inputs)
        x_u = torch.matmul(x_u, self.weights_u)

        x_v = self.dropout(item_inputs)
        x_v = torch.matmul(x_v, self.weights_v)

        u_outputs = self.activation(x_u)
        v_outputs = self.activation(x_v)

        if self.use_bias:
            u_outputs += self.user_bias
            v_outputs += self.item_bias

        return u_outputs, v_outputs


class Decoder(nn.Module):
    def __init__(self, input_dim, num_weights, num_classes, dropout=0.,activation=lambda x:x):
        """解码器
        
        Args:
        ----
            input_dim (int): 输入的特征维度
            num_weights (int): basis weight number
            num_classes (int): 总共的评分级别数，eg. 5
        """
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_weights = num_weights
        self.num_classes = num_classes
        self.activation = activation

        
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(input_dim, input_dim))
                                        for _ in range(num_weights)])
        self.weight_classifier = nn.Parameter(torch.Tensor(num_weights, num_classes))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.weight)):
            init.orthogonal_(self.weight[i], gain=1.1)
        init.xavier_uniform_(self.weight_classifier)

    def forward(self, user_inputs, item_inputs, user_indices, item_indices):
        """计算非归一化的分类输出
        
        Args:
            user_inputs (torch.Tensor): 用户的隐层特征
            item_inputs (torch.Tensor): 商品的隐层特征
            user_indices (torch.LongTensor): 
                所有交互行为中用户的id索引，与对应的item_indices构成一条边,shape=(num_edges, )
            item_indices (torch.LongTensor): 
                所有交互行为中商品的id索引，与对应的user_indices构成一条边,shape=(num_edges, )
        
        Returns:
            [torch.Tensor]: 未归一化的分类输出，shape=(num_edges, num_classes)
        """
        user_inputs = self.dropout(user_inputs)
        item_inputs = self.dropout(item_inputs)
        user_inputs = user_inputs[user_indices]
        item_inputs = item_inputs[item_indices]
        
        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(user_inputs, self.weight[i])
            out = torch.sum(tmp * item_inputs, dim=1, keepdim=True)
            basis_outputs.append(out)

        basis_outputs = torch.cat(basis_outputs, dim=1)
        
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)
        
        return outputs
