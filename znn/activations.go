package znn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// sigmoid squishification function
func Sigmoid(num float64) float64 {
	return 1.0 / (1.0 + math.Exp(-num))
}

// derivative of sigmoid for backprop
func SigmoidPrime(num float64) float64 {
	return Sigmoid(num) * (1.0 - Sigmoid(num))
}

func ApplySigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return Mult(m, Subtract(ones, m)) // m * (1 - m)
}

// (Rectified Linear Units) returns 0 if negative, or the number
func ReLU(num float64) float64 {
	return math.Max(0.0, num)
}

// derivative of ReLU for backprop
func ReLUPrime(num float64) float64 {
	if num < 0.0 {
		return 0.0
	}
	return 1.0
}

func ApplyReLUPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		if o[i] < 0.0 {
			o[i] = 0.0
		} else {
			o[i] = 1.0
		}
	}
	d := mat.NewDense(rows, 1, o)
	return d
}