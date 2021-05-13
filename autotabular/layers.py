import torch
import torch.nn as nn
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
        fm_input = inputs
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
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