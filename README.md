## Clothing Classification Using Convolutional Neural Nets
Curious to see how Data Science can be used in the fashion industry, I found an MNIST-like fashion dataset that I trained a Convolutional Neural Net(CNN) to solve a multi-clothing image classification problem. The dataset consisted of 10 different articles of clothing with each image being 28 x 28 pixels and gray scaled. The CNN I trained was a 3-layer sequential model that used a ReLU actvation function. I also included 2 dropout layers at a rate of .25 which highly improved the validation accuracy. However in order to lower the validation accuracy below the .2 threshhold, I implemented one of Keras' callback functions, ReducedLROnPlateau to slow the learning rate when no improvement was seen. This further icreased the validation accuracy however at the expense of the model overfitting and not improving around the 30th epoch.


The dataset can be downloaded at:

https://www.kaggle.com/zalando-research/fashionmnist
