package main

import (
	"fmt"

	"github.com/sno6/nett"
)

const (
	inputNodes  = 2
	outputNodes = 1
	hiddenNodes = 10
)

func main() {
	trainingData := []*nett.TrainingSample{
		{Input: nett.NewMatrixFromSlice([]float64{1, 0}), Output: nett.NewMatrixFromSlice([]float64{0})},
		{Input: nett.NewMatrixFromSlice([]float64{0, 1}), Output: nett.NewMatrixFromSlice([]float64{0})},
		{Input: nett.NewMatrixFromSlice([]float64{0, 0}), Output: nett.NewMatrixFromSlice([]float64{0})},
		{Input: nett.NewMatrixFromSlice([]float64{1, 1}), Output: nett.NewMatrixFromSlice([]float64{1})},
	}

	// Create a new nett network that will be responsible for training our model.
	n := nett.New(&nett.TrainingConfig{
		LearningRate: 0.4,
		Epochs:       100000,
		LossFunc:     nett.EuclideanLoss,
	})

	// Initialise the model layers that we want to train. Note here that input and output layers are included.
	n.InitModel(
		nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: inputNodes, Activation: nett.Sigmoid}),
		nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: hiddenNodes, Activation: nett.Sigmoid}),
		nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: hiddenNodes, Activation: nett.Sigmoid}),
		nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: outputNodes, Activation: nett.Sigmoid}),
	)

	// Pass our training data through the network.
	n.Train(trainingData)

	// Test how well our network did.
	for _, t := range trainingData {
		out := n.Forward(t.Input)
		fmt.Printf("Given the input: %v, the network predicted -> %v\n", t.Input.Pretty(), out.Pretty())
	}
}
