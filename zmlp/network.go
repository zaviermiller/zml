package zmlp

import (
	"math"
	"math/rand"
	"time"
	"fmt"
	"errors"
	"strings"
	"os"


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
	// configuration for the network
	config 		  NNConfig

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

	return nn
}

// training function using backpropogation
func (nn *NeuralNetwork) Train(inputArr, expected [][]float64, setSize int) error {
	
	for e := 0; e < nn.config.NumEpochs; e++ {
		fmt.Println(fmt.Sprintf("Epoch #%d", e + 1))
		for s := 0; s < setSize; s++ {
			printProgress(float64(s) / float64(setSize), 50.0)
			// forward propagation
			inputs := mat.NewDense(len(inputArr[s]), 1, inputArr[s])
			hiddenInputs := Dot(nn.hiddenWeights, inputs)
			hiddenOutputs := Apply(applyActivation, hiddenInputs)
			finalInputs := Dot(nn.outputWeights, hiddenOutputs)
			finalOutputs := Apply(applyActivation, finalInputs)

			// find loss
			targets := mat.NewDense(len(expected[s]), 1, expected[s])
			outputLosses := Subtract(targets, finalOutputs)
			hiddenLosses := Dot(nn.outputWeights.T(), outputLosses)

			// backpropagate
			nn.outputWeights = Add(nn.outputWeights,
				Scale(nn.config.LearningRate,
					Dot(Mult(outputLosses, ApplySigmoidPrime(finalOutputs)),
						hiddenOutputs.T()))).(*mat.Dense)
			
			nn.hiddenWeights = Add(nn.hiddenWeights,
				Scale(nn.config.LearningRate,
					Dot(Mult(hiddenLosses, ApplySigmoidPrime(hiddenOutputs)),
						inputs.T()))).(*mat.Dense)
		}
	}
	
	return nil
}

// predict using the trained model feeding forward
func (nn *NeuralNetwork) Predict(inputData []float64) (mat.Matrix, error) {

	// make sure model is trained
	if nn.hiddenWeights == nil || nn.outputWeights == nil {
		return nil, errors.New("UNTRAINED MODEL: No weights")
	}
	// if nn.hiddenBias == nil || nn.outputBias == nil {
	// 	return nil, errors.New("UNTRAINED MODEL: No biases")
	// }

	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := Dot(nn.hiddenWeights, inputs)
	hiddenOutputs := Apply(applyActivation, hiddenInputs)
	finalInputs := Dot(nn.outputWeights, hiddenOutputs)
	finalOutputs := Apply(applyActivation, finalInputs)

	return finalOutputs, nil

}

// private

func printProgress(progress float64, width float64) {
	equals := strings.Repeat("=", int(progress * width))
	dashes := strings.Repeat("-", int((1.0 - progress) * width))
	fmt.Print("[ " + equals + dashes + " ]")
	if (progress != 1.0) {
		fmt.Print("\r")
	} else {
		fmt.Println()
	}
}

// basic backprop algorithm from blogpost **INEFFICIENT**
func (nn *NeuralNetwork) backprop(inputs, expected, hiddenWeights, hiddenBias, outputWeights, outputBias, output *mat.Dense) error {

	for i := 0; i < nn.config.NumEpochs; i++ {

		// feed forward prediction
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(inputs, hiddenWeights)
		addHiddenBias := func(_, col int, val float64) float64 { return val + hiddenBias.At(0, col) }
		hiddenLayerInput.Apply(addHiddenBias, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		hiddenLayerActivations.Apply(applyActivation, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, outputWeights)
		addOutputBias := func(_, col int, val float64) float64 { return val + outputBias.At(0, col) }
		outputLayerInput.Apply(addOutputBias, outputLayerInput)

		output.Apply(applyActivation, outputLayerInput)

		// find loss
		fmt.Println(output)
		loss := new(mat.Dense)
		loss.Sub(expected, output)
		
		// backpropagate
		slopeOutputLayer := new(mat.Dense)
		slopeOutputLayer.Apply(applyActivationPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applyActivationPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(loss, slopeOutputLayer)
		hiddenLayerLoss := new(mat.Dense)
		hiddenLayerLoss.Mul(dOutput, outputWeights.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(hiddenLayerLoss, slopeHiddenLayer)

		// adjust params
		outputWeightsAdjust := new(mat.Dense)
		outputWeightsAdjust.Mul(hiddenLayerActivations.T(), dOutput)
		outputWeightsAdjust.Scale(nn.config.LearningRate, outputWeightsAdjust)
		outputWeights.Add(outputWeights, outputWeightsAdjust)
		outputBiasAdjust, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		outputBiasAdjust.Scale(nn.config.LearningRate, outputBiasAdjust)
		outputBias.Add(outputBias, outputBiasAdjust)
		
		hiddenWeightsAdjust := new(mat.Dense)
		hiddenWeightsAdjust.Mul(inputs.T(), dHiddenLayer)
		hiddenWeightsAdjust.Scale(nn.config.LearningRate, hiddenWeightsAdjust)
		hiddenWeights.Add(hiddenWeights, hiddenWeightsAdjust)
		hiddenBiasAdjust, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		hiddenBiasAdjust.Scale(nn.config.LearningRate, hiddenBiasAdjust)
		hiddenBias.Add(hiddenBias, hiddenBiasAdjust)
	}

	return nil
}

func (nn *NeuralNetwork) Save() {
	h, err := os.Create("data/zmlp-hiddem.model")
	defer h.Close()
	if err == nil {
		nn.hiddenWeights.MarshalBinaryTo(h)
		fmt.Println("Saved hidden weights")
	}
	o, err := os.Create("data/zmlp-output.model")
	defer o.Close()
	if err == nil {
		nn.outputWeights.MarshalBinaryTo(o)
		fmt.Println("saved output weights")
	}
}

// load a neural network from file
func (nn *NeuralNetwork) Load() {
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}

// sumAlongAxis sums a matrix along a particular dimension, 
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

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
			return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
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