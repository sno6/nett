package main

import (
	"fmt"

	"github.com/sno6/nett"
)

func main() {
	trainingData := []*nett.TrainingSample{
		{Input: nett.NewMatrixFromSlice([]float64{1, 0}), Output: nett.NewMatrixFromSlice([]float64{0})},
		{Input: nett.NewMatrixFromSlice([]float64{0, 1}), Output: nett.NewMatrixFromSlice([]float64{0})},
		{Input: nett.NewMatrixFromSlice([]float64{0, 0}), Output: nett.NewMatrixFromSlice([]float64{0})},
		{Input: nett.NewMatrixFromSlice([]float64{1, 1}), Output: nett.NewMatrixFromSlice([]float64{1})},
	}

	// Train our neural network to learn how to before the binary AND operation.
	n := nett.New(nil)
	n.Train(trainingData)

	// Let's test how well it did.
	for _, t := range trainingData {
		out := n.Forward(t.Input).At(0, 0)
		fmt.Printf("Given the input: %v, the network predicted -> %v\n", t.Input.Pretty(), out)
	}
}
