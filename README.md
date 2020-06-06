# Variational Graph Auto-encoder in Pytorch Geometric

This respository implements variational graph auto-encoder in [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric), adapted from the autoencoder example [code](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py)  in pyG. For details of the model, refer to Thomas Klpf's original [paper](https://arxiv.org/abs/1611.07308).

## Requirements

- Python >= 3.6
- Pytorch == 1.5
- Pytorch Geometric == 1.5
- scikit-learn
- scipy

## How to run

1. Configure the arguments in `config/vgae.yaml` file. You can also make your own config file.

2. Specify the config file and run the training script.
```
python train.py --load_config config/vgae.yaml
```