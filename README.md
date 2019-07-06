# Associated Learning

Associated Learning, which modularizes neural network into smaller components, each of which has a local objective. 
Because the objectives are mutually independent, Associated Learning can learn the parameters independently and simultaneously when these parameters belong to different components. 
Each folder contains a main.py and a models.py. The folder name corresponds to the experimental dataset.

See the following paper for more background:

[1] [Associated Learning: Decomposing End-to-end Backpropagation based on Auto-encoders and Target Propagation](https://arxiv.org/pdf/1906.05560.pdf) 
by Yu-Wei Kao and Hung-Hsuan Chen

## Tested Environment
* Windows 10 (1903)
* Tensorflow 1.13.1 and 1.14.0

## Usage
Specifying the model you want to test with `--model` flag and run the following:
```bash
python main.py --model <MODEL>
```

For MNIST dataset, there are 4 models you can choose:
* MLP
* MLP_AL
* CNN
* CNN_AL

For CIFAR-10 and CIFAR-100, there are 10 models you can choose:
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
