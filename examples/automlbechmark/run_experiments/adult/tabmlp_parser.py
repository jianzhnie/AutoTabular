import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='TabMlp parameters')

    # model parameters
    parser.add_argument(
        '--mlp_hidden_dims',
        type=str,
        default='[200, 100]',
        help='if auto it will do 4 x inp_dim -> 2 x inp_dim -> out',
    )
    parser.add_argument(
        '--mlp_activation',
        type=str,
        default='relu',
        help='one of relu, leaky_relu, gelu ',
    )
    parser.add_argument(
        '--mlp_dropout', type=float, default=0.1, help='mlp dropout')
    parser.add_argument(
        '--mlp_batchnorm',
        action='store_true',
        help='if true the dense layers will be built with BatchNorm',
    )
    parser.add_argument(
        '--mlp_batchnorm_last',
        action='store_true',
        help=
        'if true BatchNorm will be applied to the last of the dense layers',
    )
    parser.add_argument(
        '--mlp_linear_first',
        action='store_true',
        help='Boolean indicating the order of the operations in the dense',
    )
    parser.add_argument(
        '--embed_dropout', type=float, default=0.0, help='embeddings dropout')

    # train/eval parameters
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument(
        '--n_epochs', type=int, default=1, help='Number of epoch.')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument(
        '--weight_decay', type=float, default=0.0, help='l2 reg.')
    parser.add_argument(
        '--eval_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument(
        '--early_stop_delta',
        type=float,
        default=0.0,
        help='Min delta for early stopping',
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=20,
        help='Patience for early stopping',
    )
    parser.add_argument(
        '--monitor',
        type=str,
        default='val_loss',
        help='(val_)loss or (val_)metric name to monitor',
    )

    # Optimizer parameters
    parser.add_argument(
        '--optimizer',
        type=str,
        default='UseDefault',
        help=
        'Only Adam, AdamW, and RAdam are considered. UseDefault is AdamW with default values',
    )

    # Scheduler parameters
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='ReduceLROnPlateau',
        help=
        "one of 'ReduceLROnPlateau', 'CyclicLR' or 'OneCycleLR', NoScheduler",
    )
    # ReduceLROnPlateau (rop) params
    parser.add_argument(
        '--rop_mode',
        type=str,
        default='min',
        help='One of min, max',
    )
    parser.add_argument(
        '--rop_factor',
        type=float,
        default=0.1,
        help='Factor by which the learning rate will be reduced',
    )
    parser.add_argument(
        '--rop_patience',
        type=int,
        default=10,
        help=
        'Number of epochs with no improvement after which learning rate will be reduced',
    )
    parser.add_argument(
        '--rop_threshold',
        type=float,
        default=0.001,
        help='Threshold for measuring the new optimum',
    )
    parser.add_argument(
        '--rop_threshold_mode',
        type=str,
        default='abs',
        help='One of rel, abs',
    )
    # CyclicLR and OneCycleLR params
    parser.add_argument(
        '--base_lr',
        type=float,
        default=0.001,
        help='base_lr for cyclic lr_schedulers',
    )
    parser.add_argument(
        '--max_lr',
        type=float,
        default=0.01,
        help='max_lr for cyclic lr_schedulers',
    )
    parser.add_argument(
        '--div_factor',
        type=float,
        default=25,
        help=
        'Determines the initial learning rate via initial_lr = max_lr/div_factor',
    )
    parser.add_argument(
        '--final_div_factor',
        type=float,
        default=1e4,
        help=
        'Determines the minimum learning rate via min_lr = initial_lr/final_div_factor',
    )
    parser.add_argument(
        '--n_cycles',
        type=float,
        default=5,
        help='number of cycles for CyclicLR',
    )
    parser.add_argument(
        '--cycle_momentum',
        action='store_true',
    )
    parser.add_argument(
        '--pct_step_up',
        type=float,
        default=0.3,
        help=
        'Percentage of the cycle (in number of steps) spent increasing the learning rate',
    )

    # save parameters
    parser.add_argument(
        '--save_results', action='store_true', help='Save model and results')

    return parser.parse_args()
