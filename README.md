# ZML
For at least my first neural network experience, I want to build a simple MLP with backpropogation from the ground up, using the calculus and matrix algebra required. 
I may end up doing this for other NNs (CNN, RNN, CRNN) but for now, just the MLP

### ZNN
The most basic neural net with an input layer, hidden layer, and output layer. This structure actually works best for the MNIST number set, getting around 97% when not running training concurrently.

### ZDNN
A more complex network with variable, configurable hidden layers. Running more than 1 layer gives worse and worse performance on the MNIST set, but I may test it on some other data as well.