import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class FM(nn.Module):
    """Factorization Machine to model order-2 feature interactions
    Arguments:

    Call arguments:
        inputs: A 3D tensor.

    Input shape
    -----------
        - 3D tensor with shape:
         `(batch_size, field_size, embedding_size)`

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, 1)`

    References
    ----------
    .. [1] `Rendle S. Factorization machines[C]//2010 IEEE International Conference on Data Mining. IEEE, 2010: 995-1000.`
    .. [2] `Guo H, Tang R, Ye Y, et al. Deepfm: An end-to-end wide & deep learning framework for CTR prediction[J]. arXiv preprint arXiv:1804.04950, 2018.`
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {len(inputs.shape)}.')

        square_of_sum = torch.pow(torch.sum(inputs, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(inputs * inputs, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class SENETLayer(nn.Module):
    """SENET layer can dynamically increase the weights of important features and decrease the weights
    of uninformative features to let the model pay more attention to more important features.

        Arguments:
            filed_size : int, Positive integer, number of feature groups.
            pooling_op: str, (default='mean')
                pooling methods to squeeze the original embedding E into a statistic vector Z
                - mean
                - max
            reduction_ratio: float, (default=3)
                hyper-parameter for dimensionality-reduction

        Call arguments:
            inputs: A 3D tensor.

        Input shape
        -----------
            - 3D tensor with shape:
             `(batch_size, field_size, embedding_size)`

        Output shape
        ------------
            - 3D tensor with shape:
             `(batch_size, field_size, embedding_size)`

        References
        ----------
        .. [1] `Huang T, Zhang Z, Zhang J. FiBiNET: combining feature importance and bilinear feature
        interaction for click-through rate prediction[C]//Proceedings of the 13th ACM Conference on
        Recommender Systems. 2019: 169-177.`
    """

    def __init__(self, filed_size, pooling_op='mean', reduction_ratio=3, seed=1024, device='cpu'):
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.pooling_op = pooling_op
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        if self.pooling_op == 'max':
            Z = torch.max(inputs, dim=-1, out=None)
        else:
            Z = torch.mean(inputs, axis=-1, out=None)

        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))

        return V


class BilinearInteractionPooling(nn.Module):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """

    def __init__(self):
        super(BilinearInteractionPooling, self).__init__()

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term


class BilinearInteraction(nn.Module):
    """The Bilinear-Interaction layer combines the inner product and Hadamard product to learn the
    feature interactions.

    Arguments:
        iled_size : Positive integer, number of feature groups.
        embedding_size : Positive integer, embedding size of sparse features.
        bilinear_type: str, (default='field_interaction')
            the type of bilinear functions
            - field_interaction
            - field_all
            - field_each
        seed : A Python integer to use as random seed.

    Call arguments:
        inputs: A 3D tensor.

    Input shape
    -----------
        - 3D tensor with shape:
         `(batch_size, field_size, embedding_size)`

    Output shape
    ------------
        - 3D tensor with shape:
         `(batch_size, filed_size*(filed_size-1)/2, embedding_size)`

    References
    ----------
    .. [1] `Huang T, Zhang Z, Zhang J. FiBiNET: combining feature importance and bilinear feature
    interaction for click-through rate prediction[C]//Proceedings of the 13th ACM Conference on
    Recommender Systems. 2019: 169-177.`
    """

    def __init__(self, filed_size, embedding_size, bilinear_type="interaction", seed=1024, device='cpu'):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == "each":
            for _ in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == "interaction":
            for i, j in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class AFMLayer(nn.Module):
    """Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.
        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.
        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.
        - **seed** : A Python integer to use as random seed.
      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """
    def __init___(self, in_features, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, device='cpu'):
        super(AFMLayer, self).__init__()

        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_features

        self.attention_W = nn.Parameter(torch.Tensor(
            embedding_size, self.attention_factor))

        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))

        self.projection_h = nn.Parameter(
            torch.Tensor(self.attention_factor, 1))

        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor, )

        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor, )

        self.dropout = nn.Dropout(dropout_rate)

        self.to(device)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)
        
        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)

        inner_product = p * q
        bi_interaction = inner_product
        
        attention_temp = F.relu(torch.tensordot(
            bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b
        )
        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])), dim=1
        )
        attention_output = torch.sum(
            self.normalized_att_score * bi_interaction, dim=1)
        
        attention_output = self.dropout(attention_output)  # training
        
        afm_out = torch.tensordot(
            attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out


class AFM(nn.Module):
    """Attentional Factorization Machine (AFM), which learns the importance of each feature interaction
    from datasets via a neural attention network.

    Arguments:
        hidden_factor: int, (default=16)
        activation_function : str, (default='relu')
        kernel_regularizer : str or object, (default=None)
        dropout_rate: float, (default=0)

    Call arguments:
        x: A list of 3D tensor.

    Input shape
    -----------
        - A list of 3D tensor with shape: (batch_size, 1, embedding_size)

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, 1)`

    References
    ----------
    .. [1] `Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature
    interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.`
    .. [2] https://github.com/hexiangnan/attentional_factorization_machine
    """

    def __init__(self, in_features, attention_factor, dropout_rate, activation, kernel_regularizer, **kwargs):
        self.embedding_size = in_features
        self.attention_factor = attention_factor
        self.dropout_rate = dropout_rate
        self.activation_function = activation
        self.kernel_regularizer = kernel_regularizer
        super(AFM, self).__init__(**kwargs)

        self.dense_attention = nn.Linear(self.embedding_size, self.attention_factor)
        self.projection = nn.Linear(self.attention_factor, 1)
        self.fc = nn.Linear(self.embedding_size, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        row, col = list(), list()
        for r, c in itertools.combinations(inputs, 2):
            row.append(r)
            col.append(c)
        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q
        
        attention_score = nn.ReLU(self.dense_attention(inner_product))
        attention_score = nn.Softmax(self.projection(attention_score), dim=1)

        attention_out = torch.sum(attention_score * inner_product, dim=1)
        attention_out = self.dropout(attention_out)
        afm_out = self.fc(attention_out)
        return afm_out


class InnerProductLayer(nn.Module):
    """Inner-Product layer

    Arguments:
        reduce_sum: bool. Whether return inner product or element-wise product

    Call arguments:
        Inputs: A list of 3D tensor.

    Input shape
    -----------
        - A list of 3D tensor with shape (batch_size, 1, embedding_size)

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, num_fields*(num_fields-1)/2)`

    References
    ----------
    .. [1] `Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//2016
    IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016: 1149-1154.`
    .. [2] `Qu Y, Fang B, Zhang W, et al. Product-based neural networks for user response prediction over
    multi-field categorical datasets[J]. ACM Transactions on Information Systems (TOIS), 2018, 37(1): 1-35.`
    .. [3] https://github.com/Atomu2014/product-nets
    """

    def __init__(self, reduce_sum=True, device='cpu'):
        super(InnerProductLayer, self).__init__()
        self.reduce_sum = reduce_sum
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx]
                       for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx]
                       for idx in col], dim=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = torch.sum(
                inner_product, dim=2, keepdim=True)
        return inner_product


class OutterProductLayer(nn.Module):
    """Outer-Product layer

    Arguments:
        filed_size: Positive integer, number of feature groups.
        embedding_size: embedding_size
        kernel_type: str, (default='mat')
            the type of outer product kernel
            - mat
            - vec
            - num

    Call arguments:
        x: A list of 3D tensor.

    Input shape
    -----------
        - A list of 3D tensor with shape (batch_size, 1, embedding_size)

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, num_fields*(num_fields-1)/2)`

    References
    ----------
    .. [1] `Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//2016
    IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016: 1149-1154.`
    .. [2] `Qu Y, Fang B, Zhang W, et al. Product-based neural networks for user response prediction over
    multi-field categorical datasets[J]. ACM Transactions on Information Systems (TOIS), 2018, 37(1): 1-35.`
    .. [3] https://github.com/Atomu2014/product-nets
    """

    def __init__(self, field_size, embedding_size, kernel_type='mat', seed=1024, device='cpu'):
        super(OutterProductLayer, self).__init__()
        self.kernel_type = kernel_type
        if self.kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")

        num_inputs = field_size
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = embedding_size
        if self.kernel_type == 'mat':

            self.kernel = nn.Parameter(torch.Tensor(
                embed_size, num_pairs, embed_size))

        elif self.kernel_type == 'vec':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, embed_size))

        elif self.kernel_type == 'num':
            self.kernel = nn.Parameter(torch.Tensor(num_pairs, 1))
        nn.init.xavier_uniform_(self.kernel)

        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = torch.cat([embed_list[idx]
                       for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        # -------------------------
        if self.kernel_type == 'mat':
            p.unsqueeze_(dim=1)
            # k     k* pair* k
            # batch * pair
            kp = torch.sum(
                # batch * pair * k
                torch.mul(
                    # batch * pair * k
                    torch.transpose(
                        # batch * k * pair
                        torch.sum(
                            # batch * k * pair * k
                            torch.mul(
                                p, self.kernel),
                            dim=-1),
                        2, 1),
                    q),
                dim=-1)
        else:
            # 1 * pair * (k or 1)
            k = torch.unsqueeze(self.kernel, 0)
            # batch * pair
            kp = torch.sum(p * q * k, dim=-1)
            # p q # b * p * k
        return kp