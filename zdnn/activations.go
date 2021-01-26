package zdnn

import (
	"math"

	"gonum.org/v1/gonum/mat"
	// "fmt"
)

type Activation int

const (
	Sigmoid Activation = iota
	ReLU
	Softmax
)

type IActivation interface {
	Apply(mat.Matrix) mat.Matrix
	ApplyPrime(mat.Matrix) mat.Matrix
}

type SigmoidStruct struct{}
type ReLUStruct struct{}
type SoftmaxStruct struct{}

func NewActivation(opt Activation) IActivation {
	switch opt {
	case Sigmoid:
		return SigmoidStruct{}
	case ReLU:
		return ReLUStruct{}
		// case Softmax:
		// 	return SoftmaxStruct{}
	}
	return nil
}

func (s SigmoidStruct) Apply(m mat.Matrix) mat.Matrix {
	apply := func(_, _ int, val float64) float64 { return sigmoid(val) }
	return Apply(apply, m)
}

// sigmoid squishification function
func sigmoid(num float64) float64 {
	return 1.0 / (1.0 + math.Exp(-num))
}

func (s SigmoidStruct) ApplyPrime(m mat.Matrix) mat.Matrix {
	apply := func(_, _ int, val float64) float64 { return sigmoidPrime(val) }
	return Apply(apply, m)
}

// derivative of sigmoid for backprop
func sigmoidPrime(num float64) float64 {
	return sigmoid(num) * (1.0 - sigmoid(num))
}

func (r ReLUStruct) Apply(m mat.Matrix) mat.Matrix {
	apply := func(_, _ int, val float64) float64 { return relu(val) }
	return Apply(apply, m)
}

// (Rectified Linear Units) returns 0 if negative, or the number
func relu(num float64) float64 {
	return math.Max(0.0, num)
}

func (r ReLUStruct) ApplyPrime(m mat.Matrix) mat.Matrix {
	apply := func(_, _ int, val float64) float64 { return reluPrime(val) }
	return Apply(apply, m)
}

// derivative of ReLU for backprop
func reluPrime(num float64) float64 {
	if num < 0.0 {
		return 0.0
	}
	return 1.0
}

// func (s SoftmaxStruct) Apply(m mat.Matrix) mat.Matrix {
// 	max := mat.Max(m)
// 	eXMat := Apply(func(_, _ int, val float64) float64 { return math.Exp(val - max) }, m)
// 	apply := func(_, _ int, val float64) float64 { return softmax(val, mat.Max(eXMat)) }
// 	return Apply(apply, eXMat)
// }

// func softmax(val, sum float64) float64 {
// 	return val / sum
// }

// func (s SoftmaxStruct) ApplyPrime(m mat.Matrix) mat.Matrix {
// 	apply := func(_, _ int, val float64) float64 { return reluPrime(val) }
// 	return Apply(apply, m)
// }
