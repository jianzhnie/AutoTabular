from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP
from pytorch_widedeep.models.transformers._encoders import FastFormerEncoder
from pytorch_widedeep.models.transformers._embeddings_layers import (
    CatAndContEmbeddings,
)


class TabFastFormer(nn.Module):
    r"""Defines an adaptation of a ``FastFormer`` model
    (`arXiv:2108.09084 <https://arxiv.org/abs/2108.09084>`_) that can be used
    as the ``deeptabular`` component of a Wide & Deep model.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        {'education': 0, 'relationship': 1, 'workclass': 2, ...}
    embed_input: List
        List of Tuples with the column name and number of unique values
        e.g. [('education', 11), ...]
    embed_dropout: float, default = 0.1
        Dropout to be applied to the embeddings matrix
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        :obj:`pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If ``full_embed_dropout = True``, ``embed_dropout`` is ignored.
    shared_embed: bool, default = False
        The idea behind ``shared_embed`` is described in the Appendix A in the
        `TabTransformer paper <https://arxiv.org/abs/2012.06678>`_: `'The
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns'`. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        ``frac_shared_embed`` with the shared embeddings.
        See :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if ``add_shared_embed
        = False``) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    embed_continuous_activation: str, default = None
        String indicating the activation function to be applied to the
        continuous embeddings, if any. ``tanh``, ``relu``, ``leaky_relu`` and
        ``gelu`` are supported.
    cont_norm_layer: str, default =  None,
        Type of normalization layer applied to the continuous features before
        they are embedded. Options are: ``layernorm``, ``batchnorm`` or
        ``None``.
    input_dim: int, default = 32
        The so-called *dimension of the model*. In general is the number of
        embeddings used to encode the categorical and/or continuous columns
    n_heads: int, default = 8
        Number of attention heads per FastFormer block
    use_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers
    n_blocks: int, default = 4
        Number of FastFormer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Additive Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    share_qv_weights: bool, default = False
        Following the paper, this is a boolean indicating if the the value and
        the query transformation parameters will be shared
    share_weights: bool, default = False
        In addition to sharing the value and query transformation parameters,
        the parameters across different Fastformer layers can also be shared
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. ``tanh``, ``relu``,
        ``leaky_relu``, ``gelu``, ``geglu`` and ``reglu`` are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to ``[l, 4*l,
        2*l]`` where ``l`` is the MLP input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. ``tanh``, ``relu``, ``leaky_relu`` and
        ``gelu`` are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``

    Attributes
    ----------
    cat_and_cont_embed: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    transformer_blks: ``nn.Sequential``
        Sequence of FasFormer blocks.
    transformer_mlp: ``nn.Module``
        MLP component in the model
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabFastFormer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabFastFormer(column_idx=column_idx, embed_input=embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int]]] = None,
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous_activation: str = None,
        cont_norm_layer: str = None,
        input_dim: int = 32,
        n_heads: int = 8,
        use_bias: bool = False,
        n_blocks: int = 4,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.2,
        share_qv_weights: bool = False,
        share_weights: bool = False,
        transformer_activation: str = "relu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(TabFastFormer, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_dropout = embed_dropout
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.continuous_cols = continuous_cols
        self.embed_continuous_activation = embed_continuous_activation
        self.cont_norm_layer = cont_norm_layer
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.use_bias = use_bias
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.share_qv_weights = share_qv_weights
        self.share_weights = share_weights
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.with_cls_token = "cls_token" in column_idx
        self.n_cat = len(embed_input) if embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.n_feats = self.n_cat + self.n_cont

        self.cat_and_cont_embed = CatAndContEmbeddings(
            input_dim,
            column_idx,
            embed_input,
            embed_dropout,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            False,  # use_embed_bias
            continuous_cols,
            True,  # embed_continuous,
            embed_continuous_activation,
            True,  # use_cont_bias
            cont_norm_layer,
        )

        self.transformer_blks = nn.Sequential()
        first_fastformer_block = FastFormerEncoder(
            input_dim,
            n_heads,
            use_bias,
            attn_dropout,
            ff_dropout,
            share_qv_weights,
            transformer_activation,
        )
        self.transformer_blks.add_module("fastformer_block0", first_fastformer_block)
        for i in range(1, n_blocks):
            if share_weights:
                self.transformer_blks.add_module(
                    "fastformer_block" + str(i), first_fastformer_block
                )
            else:
                self.transformer_blks.add_module(
                    "fastformer_block" + str(i),
                    FastFormerEncoder(
                        input_dim,
                        n_heads,
                        use_bias,
                        attn_dropout,
                        ff_dropout,
                        share_qv_weights,
                        transformer_activation,
                    ),
                )

        attn_output_dim = (
            self.input_dim
            if self.with_cls_token
            else (self.n_cat + self.n_cont) * self.input_dim
        )
        if not mlp_hidden_dims:
            mlp_hidden_dims = [
                attn_output_dim,
                attn_output_dim * 4,
                attn_output_dim * 2,
            ]
        else:
            assert mlp_hidden_dims[0] == attn_output_dim, (
                f"The input dim of the MLP must be {attn_output_dim}. "
                f"Got {mlp_hidden_dims[0]} instead"
            )
        self.transformer_mlp = MLP(
            mlp_hidden_dims,
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:

        x_cat, x_cont = self.cat_and_cont_embed(X)

        if x_cat is not None:
            x = x_cat
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont

        x = self.transformer_blks(x)

        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)

        return self.transformer_mlp(x)

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights. Each element of the list is a
        tuple where the first and second elements are :math:`\alpha`
        and :math:`\beta` attention weights in the paper.

        The shape of the attention weights is:

        :math:`(N, H, F)`

        where *N* is the batch size, *H* is the number of attention heads
        and *F* is the number of features/columns in the dataset
        """
        if self.share_weights:
            attention_weights = [self.transformer_blks[0].attn.attn_weight]
        else:
            attention_weights = [blk.attn.attn_weights for blk in self.transformer_blks]
        return attention_weights
