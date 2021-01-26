package zdnn

import (
	// "fmt"
	"gonum.org/v1/gonum/mat"
)

// NeuronLayer is a general purpose layer struct
type NeuronLayer struct {
	config LayerConfig

	activation IActivation
	weights    *mat.Dense
	bias       *mat.Dense
	output     *mat.Dense
}

// LayerConfig is the configuration for a NN Layer
type LayerConfig struct {
	Neurons    int
	Activation Activation
}

// NewLayer builds a new general purpose layer from a config object
func NewLayer(config LayerConfig) *NeuronLayer {
	act := NewActivation(config.Activation)
	return &NeuronLayer{config: config, activation: act}
}

// Update the weights and bias
func (nl *NeuronLayer) Update(weights, bias *mat.Dense) {
	nl.weights = weights
	nl.bias = bias
}
