# QMMLHackathon_CNN

Residual Network for image classification using PyTorch.

## Model Architecture:

Initial Layer: The model starts with a convolutional layer (conv0) with a 7x7 kernel, batch normalisation (bn0), ReLU activation (relu0), and a max pooling layer (maxpool0), preparing the input for the residual blocks.

Residual Blocks: Each residual block consists of two sets of convolutional layers, batch normalisation, and ReLU activations. These blocks are designed to learn residual functions with reference to the layer inputs, enabling the network to be deeper without suffering from vanishing gradients.

Dimension Matching: In some blocks, dimension matching convolutional layers (dim_match_conv) are applied to ensure the input and output dimensions align for the addition operation in the residual connections.

Global Average Pooling: Followed by the residual blocks, a global average pooling layer (global_avg_pool) reduces each feature map to a single value, decreasing the model's parameter count and making it more robust.

Output Layer: The final layer is a fully connected layer (fc1) that maps the pooled features to the class scores for the classification task.

## Initialization:

Kaiming Initialization: The model applies Kaiming initialization to all convolutional and batch normalization layers to ensure proper scaling of weights and avoid issues with deep network initialization.

## Training Setup:

Loss Function: Cross-Entropy Loss (nn.CrossEntropyLoss) is used, suitable for multi-class classification problems.

Optimizer: Stochastic Gradient Descent (SGD) with momentum and weight decay is chosen as the optimizer, with a learning rate scheduler (ReduceLROnPlateau) to adjust the learning rate based on validation loss.
