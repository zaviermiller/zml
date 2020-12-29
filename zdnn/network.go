package zdnn

import (
	"math"
	"math/rand"
	"time"
	"fmt"
	// "errors"
	"strings"
	// "os"
	"sync"


	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat/distuv"
)


var (
	applyActivation = func(_, _ int, val float64) float64 { return Sigmoid(val) }
	applyActivationPrime = func(_, _ int, val float64) float64 { return SigmoidPrime(val) }
)

// Basic NN struct to represent all the nodes
type NeuralNetwork struct {
	// configuration and mutex for the network
	config 		  NNConfig
	mu 			  sync.Mutex

	// weights and bias matrices
	hiddenLayers  []*NeuronLayer
	outputWeights *mat.Dense
	outputBias 	  *mat.Dense
}

type NeuronLayer struct {
	config  LayerConfig

	weights *mat.Dense
	bias 	*mat.Dense
}

// Simple configuration params for the network
type NNConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenLayers   []*NeuronLayer
	NumEpochs	  int
	LearningRate  float64
}

type LayerConfig struct {
	Neurons int
}

func NewLayer(config LayerConfig) *NeuronLayer {
	return &NeuronLayer { config: config }
}

// Build the network using a passed config
func NewNetwork(config NNConfig) *NeuralNetwork {
	nn := &NeuralNetwork{ config: config, hiddenLayers: []*NeuronLayer {}}

	prevSize := config.InputNeurons
	hiddenLayers := []*NeuronLayer{}
	
	for _, layer := range config.HiddenLayers {
		layer.weights = mat.NewDense(layer.config.Neurons, prevSize, randomArray(prevSize * layer.config.Neurons, float64(prevSize)))
		layer.bias = mat.NewDense(layer.config.Neurons, 1, randomArray(layer.config.Neurons, float64(layer.config.Neurons)))
		hiddenLayers = append(hiddenLayers, layer)
		prevSize = layer.config.Neurons
	}

	nn.hiddenLayers = hiddenLayers

	// init random
	rand.Seed(time.Now().UTC().UnixNano())
	nn.outputWeights = mat.NewDense(nn.config.OutputNeurons, nn.hiddenLayers[len(nn.hiddenLayers) - 1].config.Neurons, randomArray(nn.hiddenLayers[len(nn.hiddenLayers) - 1].config.Neurons * nn.config.OutputNeurons, float64(nn.hiddenLayers[len(nn.hiddenLayers) - 1].config.Neurons)))
	nn.outputBias = mat.NewDense(nn.config.OutputNeurons, 1, randomArray(nn.config.OutputNeurons, float64(nn.config.OutputNeurons)))
	
	return nn
}

// training function using backpropogation
func (nn *NeuralNetwork) Train(inputArr, expected [][]float64, setSize int, wg *sync.WaitGroup) error {
	defer wg.Done()
	fmt.Println("Started epoch")
	for s := 0; s < setSize; s++ {
		// printProgress("Epoch", float64(s), float64(setSize), 50.0)

		// forward propagation ==

		// hidden layer(s)
		inputs := mat.NewDense(len(inputArr[s]), 1, inputArr[s])
		prevLayerOutputs := inputs
		hiddenOutputs := make([]*mat.Dense, len(nn.hiddenLayers))
		for i, hiddenLayer := range nn.hiddenLayers {
			hiddenInputs := Dot(hiddenLayer.weights, prevLayerOutputs)
			hiddenOutputs[i] = Apply(applyActivation, Add(hiddenInputs, hiddenLayer.bias)).(*mat.Dense)
			prevLayerOutputs = hiddenOutputs[i]
			
		}

		// output layer
		finalInputs := Dot(nn.outputWeights, hiddenOutputs[len(hiddenOutputs) - 1])
		finalOutputs := Apply(applyActivation, Add(finalInputs, nn.outputBias))

		// find loss
		targets := mat.NewDense(len(expected[s]), 1, expected[s])
		outputLosses := Subtract(targets, finalOutputs)
		prevLosses := outputLosses
		prevWeights := nn.outputWeights
		hiddenLosses := make([]*mat.Dense, len(nn.hiddenLayers))
		for i := len(hiddenLosses) - 1; i >= 0; i-- {
			hiddenLosses[i] = Dot(prevWeights.T(), prevLosses).(*mat.Dense)
			prevWeights = nn.hiddenLayers[i].weights
			prevLosses = hiddenLosses[i]
		}

		// backpropagate
		dOutput := Mult(outputLosses, ApplySigmoidPrime(finalOutputs)).(*mat.Dense)

		outputWeights := Add(nn.outputWeights,
			Scale(nn.config.LearningRate,
				Dot(dOutput, hiddenOutputs[len(hiddenOutputs) - 1].T()))).(*mat.Dense)

		outputBias := Add(nn.outputBias,
			Scale(nn.config.LearningRate, sumAlongAxis(1, dOutput))).(*mat.Dense)
		
		hiddenLayers := make([]*NeuronLayer, len(nn.hiddenLayers))

		for i := len(hiddenLayers) - 1; i >= 0; i-- {
			var prevOut *mat.Dense
			if i == 0 {
				prevOut = inputs
			} else {
				prevOut = hiddenOutputs[i - 1]
			}

			dHidden := Mult(hiddenLosses[i], ApplySigmoidPrime(hiddenOutputs[i])).(*mat.Dense)

			hiddenWeights := Add(nn.hiddenLayers[i].weights,
				Scale(nn.config.LearningRate, Dot(dHidden, prevOut.T()))).(*mat.Dense)

			hiddenBias := Add(nn.hiddenLayers[i].bias,
				Scale(nn.config.LearningRate, sumAlongAxis(1, dHidden))).(*mat.Dense)
			
			hiddenLayers[i] = &NeuronLayer { config: nn.hiddenLayers[i].config, weights: hiddenWeights, bias: hiddenBias}
		}

		nn.mu.Lock()
		nn.outputWeights = outputWeights
		nn.outputBias = outputBias
		nn.hiddenLayers = hiddenLayers
		nn.mu.Unlock()
	}
	return nil
}

// predict using the trained model feeding forward
func (nn *NeuralNetwork) Predict(inputData []float64) (mat.Matrix, error) {


	inputs := mat.NewDense(len(inputData), 1, inputData)
	prevLayerOutputs := inputs
	hiddenOutputs := make([]*mat.Dense, len(nn.hiddenLayers))

	for i, hiddenLayer := range nn.hiddenLayers {
		hiddenInputs := Dot(hiddenLayer.weights, prevLayerOutputs)
		hiddenOutputs[i] = Apply(applyActivation, Add(hiddenInputs, hiddenLayer.bias)).(*mat.Dense)
		prevLayerOutputs = hiddenOutputs[i]
		
	}

	// output layer
	finalInputs := Dot(nn.outputWeights, hiddenOutputs[len(hiddenOutputs) - 1])
	finalOutputs := Apply(applyActivation, Add(finalInputs, nn.outputBias))

	return finalOutputs, nil

}

// private

func printProgress(label string, done float64, total float64, width float64) {
	progress := done / (total - 1)
	equals := strings.Repeat("=", int(progress * width))
	dashes := strings.Repeat("-", int((1.0 - progress) * width))
	fmt.Print(fmt.Sprintf("   " + label + " [ " + equals + dashes + " ] [%d/%d samples] - (%f", int(done), int(total), math.Round(progress * 100.0 * 100.0) / 100.0) + "%) \r")
	if (progress == 1.0) {
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

// sumAlongAxis sums a matrix along a particular dimension, 
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
			data := make([]float64, numCols)
			for i := 0; i < numCols; i++ {
					col := mat.Col(nil, i, m)
					data[i] = floats.Sum(col)
			}
			output = mat.NewDense(1, numCols, data)
	case 1:
			data := make([]float64, numRows)
			for i := 0; i < numRows; i++ {
					row := mat.Row(nil, i, m)
					data[i] = floats.Sum(row)
			}
			output = mat.NewDense(numRows, 1, data)
	default:
			return nil
	}

	return output
}

// generate random array of start weights
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}