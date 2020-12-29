package znn

import (
	"math"
	"math/rand"
	"time"
	"fmt"
	"errors"
	"strings"
	"os"
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
	hiddenWeights *mat.Dense
	hiddenBias 	  *mat.Dense
	outputWeights *mat.Dense
	outputBias 	  *mat.Dense
}

// Simple configuration params for the network
type NNConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs	  int
	LearningRate  float64
}

// Build the network using a passed config
func NewNetwork(config NNConfig) *NeuralNetwork {
	nn := &NeuralNetwork{ config: config }

	// init random
	rand.Seed(time.Now().UTC().UnixNano())
	nn.hiddenWeights = mat.NewDense(nn.config.HiddenNeurons, nn.config.InputNeurons, randomArray(nn.config.InputNeurons * nn.config.HiddenNeurons, float64(nn.config.InputNeurons)))
	nn.outputWeights = mat.NewDense(nn.config.OutputNeurons, nn.config.HiddenNeurons, randomArray(nn.config.HiddenNeurons * nn.config.OutputNeurons, float64(nn.config.HiddenNeurons)))
	nn.hiddenBias = mat.NewDense(nn.config.HiddenNeurons, 1, randomArray(nn.config.HiddenNeurons, float64(nn.config.HiddenNeurons)))
	nn.outputBias = mat.NewDense(nn.config.OutputNeurons, 1, randomArray(nn.config.OutputNeurons, float64(nn.config.OutputNeurons)))

	return nn
}

// training function using backpropogation
func (nn *NeuralNetwork) Train(inputArr, expected [][]float64, setSize int, wg *sync.WaitGroup) error {
	defer wg.Done()
	fmt.Println("Started epoch")
	for s := 0; s < setSize; s++ {
		// printProgress("Epoch", float64(s), float64(setSize), 50.0)

		// forward propagation

		// hidden layer
		inputs := mat.NewDense(len(inputArr[s]), 1, inputArr[s])
		hiddenInputs := Dot(nn.hiddenWeights, inputs)
		hiddenOutputs := Apply(applyActivation, Add(hiddenInputs, nn.hiddenBias))

		// output layer
		finalInputs := Dot(nn.outputWeights, hiddenOutputs)
		finalOutputs := Apply(applyActivation, Add(finalInputs, nn.outputBias))

		// find loss
		targets := mat.NewDense(len(expected[s]), 1, expected[s])
		outputLosses := Subtract(targets, finalOutputs)
		hiddenLosses := Dot(nn.outputWeights.T(), outputLosses)

		// backpropagate
		dOutput := Mult(outputLosses, ApplySigmoidPrime(finalOutputs)).(*mat.Dense)

		outputWeights := Add(nn.outputWeights,
			Scale(nn.config.LearningRate,
				Dot(dOutput,hiddenOutputs.T()))).(*mat.Dense)
		outputBias := Add(nn.outputBias,
			Scale(nn.config.LearningRate, sumAlongAxis(1, dOutput))).(*mat.Dense)
		
		dHidden := Mult(hiddenLosses, ApplySigmoidPrime(hiddenOutputs)).(*mat.Dense)

		hiddenWeights := Add(nn.hiddenWeights,
			Scale(nn.config.LearningRate, Dot(dHidden, inputs.T()))).(*mat.Dense)
		hiddenBias := Add(nn.hiddenBias,
			Scale(nn.config.LearningRate, sumAlongAxis(1, dHidden))).(*mat.Dense)

		nn.mu.Lock()
		nn.outputWeights = outputWeights
		nn.outputBias = outputBias
		nn.hiddenWeights = hiddenWeights
		nn.hiddenBias = hiddenBias
		nn.mu.Unlock()
	}
	return nil
}

// predict using the trained model feeding forward
func (nn *NeuralNetwork) Predict(inputData []float64) (mat.Matrix, error) {

	// make sure model is trained
	if nn.hiddenWeights == nil || nn.outputWeights == nil {
		return nil, errors.New("UNTRAINED MODEL: No weights")
	}
	if nn.hiddenBias == nil || nn.outputBias == nil {
		return nil, errors.New("UNTRAINED MODEL: No biases")
	}

	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := Dot(nn.hiddenWeights, inputs)
	hiddenOutputs := Apply(applyActivation, Add(hiddenInputs, nn.hiddenBias))
	finalInputs := Dot(nn.outputWeights, hiddenOutputs)
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

func (nn *NeuralNetwork) Save() {
	h, err := os.Create("data/zmlp-hidden-weights.model")
	defer h.Close()
	if err == nil {
		nn.hiddenWeights.MarshalBinaryTo(h)
		fmt.Println("Saved hidden weights")
	}
	o, err := os.Create("data/zmlp-output-weights.model")
	defer o.Close()
	if err == nil {
		nn.outputWeights.MarshalBinaryTo(o)
		fmt.Println("saved output weights")
	}
	o, err = os.Create("data/zmlp-output-bias.model")
	defer o.Close()
	if err == nil {
		nn.outputBias.MarshalBinaryTo(o)
		fmt.Println("saved output bias")
	}
	o, err = os.Create("data/zmlp-hidden-bias.model")
	defer o.Close()
	if err == nil {
		nn.hiddenBias.MarshalBinaryTo(o)
		fmt.Println("saved hidden bias")
	}
}

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