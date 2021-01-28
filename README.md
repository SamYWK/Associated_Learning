# Associated Learning

Associated Learning modularizes neural network into smaller components, each of which has a local objective. 
Because the objectives are mutually independent, Associated Learning can learn the parameters at different components independently and simultaneously. 

See the following paper for details:

[1] [Associated Learning: Decomposing End-to-end Backpropagation based on Auto-encoders and Target Propagation](https://arxiv.org/pdf/1906.05560.pdf) 
by Yu-Wei Kao and Hung-Hsuan Chen

# Citing the paper

Yu-Wei Kao and Hung-Hsuan Chen. "Associated Learning: Decomposing End-to-End Backpropagation Based on Autoencoders and Target Propagation." Neural Computation 33, no. 1 (2021): 174-193.

BibTeX: 
```BibTeX
@article{kao2021associated,
    title={Associated Learning: Decomposing End-to-End Backpropagation Based on Autoencoders and Target Propagation},
    author={Kao, Yu-Wei and Chen, Hung-Hsuan},
    journal={Neural Computation},
    volume={33},
    number={1},
    pages={174--193},
    year={2021},
    publisher={MIT Press}
}
```

## Tested Environment
* Windows 10 (1903)
* Tensorflow 1.13.1 and 1.14.0

## Usage
Each folder contains a `main.py` and a `models.py`. The folder name corresponds to the experimental dataset.
Specifying the model you want to test with `--model` flag and run the following:
```bash
python main.py --model <MODEL>
```

For MNIST dataset, 4 models can be selected:
* MLP
* MLP_AL
* CNN
* CNN_AL

For CIFAR-10 and CIFAR-100, 10 models can be selected:
* MLP
* MLP_AL
* CNN
* CNN_AL
* ResNet_20
* ResNet_20_AL
* ResNet_32
* ResNet_32_AL
* VGG
* VGG_AL
