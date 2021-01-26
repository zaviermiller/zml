package zdnn

import (
	"gonum.org/v1/gonum/mat"
	// "fmt"
)

type Loss int

const (
	CrossEntropy Loss = iota
	MeanSquared
)

type ILoss interface {
	Apply(m, t mat.Matrix) mat.Matrix
	ApplyPrime(m, t mat.Matrix) mat.Matrix
}

type CE struct{}
type MS struct{}

func NewLoss(opt Loss) ILoss {
	switch opt {
	case CrossEntropy:
		return CE{}
	case MeanSquared:
		return MS{}
	}
	return nil
}

func (l CE) Apply(m, t mat.Matrix) mat.Matrix {
	applyFn := func(_, _ int, val float64) float64 { return val * val }
	return Apply(applyFn, Subtract(t, m))
}

func (l CE) ApplyPrime(m, t mat.Matrix) mat.Matrix {

	dLoss := (Subtract(t, m)).(*mat.Dense)

	return dLoss
}

func (l MS) Apply(m, t mat.Matrix) mat.Matrix {
	applyFn := func(_, _ int, val float64) float64 { return val * val }
	return Apply(applyFn, Subtract(t, m))
}

func (l MS) ApplyPrime(m, t mat.Matrix) mat.Matrix {

	dLoss := (Subtract(t, m)).(*mat.Dense)

	return dLoss
}
