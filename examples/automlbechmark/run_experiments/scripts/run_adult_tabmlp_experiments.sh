python adult/adult_tabmlp.py --optimizer Adam --save_results

python adult/adult_tabmlp.py --optimizer AdamW --save_results
python adult/adult_tabmlp.py --optimizer RAdam --lr 0.03 --save_results
python adult/adult_tabmlp.py --optimizer RAdam --save_results

python adult/adult_tabmlp.py --optimizer Adam --monitor val_acc --rop_mode max --save_results
python adult/adult_tabmlp.py --optimizer AdamW --monitor val_acc --rop_mode max --save_results
python adult/adult_tabmlp.py --optimizer RAdam --lr 0.03 --monitor val_acc --rop_mode max --save_results
python adult/adult_tabmlp.py --optimizer RAdam --monitor val_acc --rop_mode max --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims auto --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200,100] --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims auto --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200,100] --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims auto --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200,100] --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims auto --mlp_dropout 0.5 --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims auto --mlp_dropout 0.5 --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims auto --mlp_dropout 0.5 --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200,100] --mlp_dropout 0.5 --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results

python adult/adult_tabmlp.py --optimizer Adam  --batch_size 512 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer Adam --batch_size 512 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --optimizer Adam  --batch_size 1024 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer Adam --batch_size 1024 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --optimizer AdamW  --batch_size 512 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer AdamW --batch_size 512 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --optimizer AdamW  --batch_size 1024 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer AdamW --batch_size 1024 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --optimizer RAdam  --batch_size 512 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer RAdam --batch_size 512 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --optimizer RAdam  --batch_size 1024 --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --optimizer RAdam --batch_size 1024 --early_stop_patience 30 --rop_factor 0.2 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --embed_dropout 0.1 --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_activation leaky_relu --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm_last --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --mlp_batchnorm_last --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_linear_first --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --mlp_linear_first --optimizer Adam --early_stop_patience 30 --rop_factor 0.2 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --embed_dropout 0.1 --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_activation leaky_relu --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm_last --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --mlp_batchnorm_last --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_linear_first --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --mlp_linear_first --optimizer AdamW --early_stop_patience 30 --rop_factor 0.2 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --embed_dropout 0.1 --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_activation leaky_relu --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm_last --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --mlp_batchnorm_last --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_linear_first --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_batchnorm --mlp_linear_first --optimizer RAdam --early_stop_patience 30 --rop_factor 0.2 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer Adam --early_stop_patience 30 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer AdamW --early_stop_patience 30 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer RAdam --early_stop_patience 30 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer Adam --early_stop_patience 30 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer AdamW --early_stop_patience 30 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer RAdam --early_stop_patience 30 --lr_scheduler CyclicLR --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer Adam --early_stop_patience 30 --lr_scheduler CyclicLR --batch_size 64 --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer AdamW --early_stop_patience 30 --lr_scheduler CyclicLR --batch_size 64 --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer RAdam --early_stop_patience 30 --lr_scheduler CyclicLR --batch_size 64 --lr 5e-4 --base_lr 5e-4 --max_lr 0.01 --n_cycles 10 --n_epochs 100 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 32 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 1 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 2 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler OneCycleLR --batch_size 64 --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 2 --save_results

python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [100,50] --mlp_dropout 0.2 --optimizer RAdam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer Adam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer AdamW --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
python adult/adult_tabmlp.py --mlp_hidden_dims [400,200] --mlp_dropout 0.5 --optimizer RAdam --lr_scheduler OneCycleLR --lr 4e-4 --max_lr 0.01 --final_div_factor 1e3 --n_epochs 5 --save_results
