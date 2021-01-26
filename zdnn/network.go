package zdnn

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	// "errors"
	"strings"
	// "os"
	"sync"

	"github.com/sethgrid/multibar"
	"gonum.org/v1/gonum/mat"
)

// NeuralNetwork is a basic struct to represent all the nodes
type NeuralNetwork struct {
	// configuration and other setup for net
	config NNConfig
	mu     sync.Mutex

	// layers and loss func
	layers   []*NeuronLayer
	lossFunc ILoss
}

// NNConfig is simple configuration params for the network
type NNConfig struct {
	InputNeurons int
	HiddenLayers []*NeuronLayer
	OutputLayer  *NeuronLayer
	NumEpochs    int
	LearningRate float64
	LossFunc     Loss
	BatchSize    int
}

// NewNetwork builds the network using a passed config
func NewNetwork(config NNConfig) *NeuralNetwork {

	// get the loss function from loss type
	loss := NewLoss(config.LossFunc)

	// init network struct
	nn := &NeuralNetwork{config: config, layers: append(append([]*NeuronLayer{}, config.HiddenLayers...), config.OutputLayer), lossFunc: loss} // lmao

	prevSize := nn.config.InputNeurons

	// randomly init layer w&b
	for _, layer := range nn.layers {
		layer.weights = mat.NewDense(layer.config.Neurons, prevSize, randomArray(prevSize*layer.config.Neurons, float64(prevSize)))
		layer.bias = mat.NewDense(layer.config.Neurons, 1, randomArray(layer.config.Neurons, float64(layer.config.Neurons)))
		prevSize = layer.config.Neurons
	}

	return nn
}

// Train the network [nn.config.NumEpochs] times (fully train the network)
func (nn *NeuralNetwork) Train(inputArr, expected [][]float64, setSize int) {
	var wg sync.WaitGroup

	// create the multibar container
	// this allows our bars to work together without stomping on one another
	progressBars, _ := multibar.New()

	for e := 0; e < nn.config.NumEpochs; e++ {
		var batchNum int = setSize / nn.config.BatchSize

		load := progressBars.MakeBar(batchNum, fmt.Sprintf("Epoch #%d", e+1))
		progressBars.Bars[e].Width = 60

		// shuffle inputs
		rand.Seed(time.Now().UnixNano())
		rand.Shuffle(setSize, func(i, j int) { inputArr[i], inputArr[j] = inputArr[j], inputArr[i] })

		go func() {
			wg.Add(1)
			for i := 0; i < batchNum; i++ {
				load(i)
				// get batch for training
				batch := inputArr[i*nn.config.BatchSize : (i+1)*nn.config.BatchSize]
				nn.TrainBatch(batch, expected, nn.config.BatchSize, &wg)
			}
		}()
	}

	go progressBars.Listen()

	wg.Wait()

}

// TrainBatch trains the network on the batch of inputs, entirely
func (nn *NeuralNetwork) TrainBatch(inputArr, expected [][]float64, setSize int, wg *sync.WaitGroup) error {
	defer wg.Done()
	var finalLayer *NeuronLayer
	if len(nn.layers) > 0 {
		finalLayer = nn.layers[len(nn.layers)-1]
	}
	for s := 0; s < setSize; s++ {
		// printProgress("Epoch", float64(s), float64(setSize), 50.0)
		// loader.PrintSimpleLoader("Epoch", stage)

		// feed forward thru nn layers
		inputs := mat.NewDense(len(inputArr[s]), 1, inputArr[s])

		// concurrent-aware forward prop
		err := nn.forwardSync(inputs)
		if err != nil {
			return err
		}

		// BACKPROP === followed https://sausheong.github.io/posts/how-to-build-a-simple-artificial-neural-network-with-go/ to learn ;]

		targets := mat.NewDense(len(expected[s]), 1, expected[s])

		// derivative of loss func with respect to the output of the last layer
		oldError := nn.lossFunc.ApplyPrime(finalLayer.output, targets).(*mat.Dense)
		dLoss := Mult(oldError, finalLayer.activation.ApplyPrime(finalLayer.output)).(*mat.Dense)

		nn.syncUpdate(func() {
			// add the scaled change to the weights
			finalWeights := Add(finalLayer.weights,
				// find d w respect to weights and scale the change by the learning rate to prevent overfit
				Scale(nn.config.LearningRate, Dot(dLoss, nn.layers[len(nn.layers)-2].output.T()))).(*mat.Dense)

			// add the scaled change to the bias
			finalBias := Add(finalLayer.bias,
				// scale dLoss by learning rate to prevent overfit
				Scale(nn.config.LearningRate, sumAlongAxis(1, dLoss))).(*mat.Dense)

			// update the values
			finalLayer.Update(finalWeights, finalBias) // may want to change this so the calcs can be outside the syncUpdate func, but if that affect concurrency
		})

		// start w/ 2nd to last bc we just did this one
		for i := len(nn.layers) - 2; i >= 0; i-- {
			layer := nn.layers[i]
			nextLayer := nn.layers[i+1]

			// previous layers outputs (may just be inputs)
			var prevOut *mat.Dense
			if i == 0 {
				prevOut = inputs
			} else {
				prevOut = nn.layers[i-1].output
			}

			// do what we just did to the final layer to the rest of 'eem
			nn.syncUpdate(func() {
				// should probably be a setter
				// layer.dLoss = Dot(prevWeights.T(), prevLosses).(*mat.Dense)
				// prevLoss := nn.layers[i].dLoss
				oldError = Dot(nextLayer.output.T(), oldError).(*mat.Dense)
				fmt.Println(oldError)
				layerLoss := Mul(oldError, layer.activation.ApplyPrime(layer.output)).(*mat.Dense)

				hiddenWeights := Add(layer.weights,
					// the hidden layers use a shortcut to find the error by doing the dot product below
					Scale(nn.config.LearningRate, Dot(layerLoss, prevOut.T()))).(*mat.Dense)

				hiddenBias := Add(layer.bias,
					Scale(nn.config.LearningRate, sumAlongAxis(1, layerLoss))).(*mat.Dense)

				layer.Update(hiddenWeights, hiddenBias)
			})
		}
	}

	return nil
}

// Predict using the trained model feeding forward
func (nn *NeuralNetwork) Predict(inputData []float64) (mat.Matrix, error) {

	prevLayerOutputs := mat.NewDense(len(inputData), 1, inputData)
	for _, layer := range nn.layers {
		hiddenInputs := Dot(layer.weights, prevLayerOutputs)
		prevLayerOutputs = layer.activation.Apply(Add(hiddenInputs, layer.bias)).(*mat.Dense)
	}

	return prevLayerOutputs, nil

}

// private

func printProgress(label string, done float64, total float64, width float64) {
	progress := done / (total - 1)
	equals := strings.Repeat("=", int(progress*width))
	dashes := strings.Repeat("-", int((1.0-progress)*width))
	fmt.Print(fmt.Sprintf("   "+label+" [ "+equals+dashes+" ] [%d/%d samples] - (%f", int(done), int(total), math.Round(progress*100.0*100.0)/100.0) + "%) \r")
	if progress == 1.0 {
		fmt.Println(label + " done!\033[K")
	}
}

// func (nn *NeuralNetwork) Save() {
// 	h, err := os.Create("data/zmlp-hidden-weights.model")
// 	defer h.Close()
// 	if err == nil {
// 		nn.hiddenWeights.MarshalBinaryTo(h)
// 		fmt.Println("Saved hidden weights")
// 	}
// 	o, err := os.Create("data/zmlp-output-weights.model")
// 	defer o.Close()
// 	if err == nil {
// 		nn.outputWeights.MarshalBinaryTo(o)
// 		fmt.Println("saved output weights")
// 	}
// 	o, err = os.Create("data/zmlp-output-bias.model")
// 	defer o.Close()
// 	if err == nil {
// 		nn.outputBias.MarshalBinaryTo(o)
// 		fmt.Println("saved output bias")
// 	}
// 	o, err = os.Create("data/zmlp-hidden-bias.model")
// 	defer o.Close()
// 	if err == nil {
// 		nn.hiddenBias.MarshalBinaryTo(o)
// 		fmt.Println("saved hidden bias")
// 	}
// }

// load a neural network from file
// func (nn *NeuralNetwork) Load() {
// 	h, err := os.Open("data/hweights.model")
// 	defer h.Close()
// 	if err == nil {
// 		net.hiddenWeights.Reset()
// 		net.hiddenWeights.UnmarshalBinaryFrom(h)
// 	}
// 	o, err := os.Open("data/oweights.model")
// 	defer o.Close()
// 	if err == nil {
// 		net.outputWeights.Reset()
// 		net.outputWeights.UnmarshalBinaryFrom(o)
// 	}
// 	return
// }

func (nn *NeuralNetwork) forwardSync(inputs *mat.Dense) error {
	prevLayerOutputs := inputs
	for _, layer := range nn.layers {
		nn.mu.Lock()
		hiddenInputs := Dot(layer.weights, prevLayerOutputs)
		layer.output = layer.activation.Apply(Add(hiddenInputs, layer.bias)).(*mat.Dense)
		prevLayerOutputs = layer.output
		nn.mu.Unlock()
	}

	return nil
}

// syncs updates thru goroutines
func (nn *NeuralNetwork) syncUpdate(updateFunc func()) error {
	nn.mu.Lock()
	updateFunc()
	nn.mu.Unlock()

	return nil
}
