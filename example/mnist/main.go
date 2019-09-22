package main

const (
	inputNodes  = 784
	hiddenNodes = 512
	outputNodes = 10
)

func main() {
	// Train a simple MLP MNIST model, 20 epochs, 784 -> 512 (RELU) -> 512 (RELU) -> 10 (Softmax)

	// trainingData := []*nett.TrainingSample{
	// 	{Input: nett.NewMatrixFromSlice([]float64{1, 0}), Output: nett.NewMatrixFromSlice([]float64{0})},
	// 	{Input: nett.NewMatrixFromSlice([]float64{0, 1}), Output: nett.NewMatrixFromSlice([]float64{0})},
	// 	{Input: nett.NewMatrixFromSlice([]float64{0, 0}), Output: nett.NewMatrixFromSlice([]float64{0})},
	// 	{Input: nett.NewMatrixFromSlice([]float64{1, 1}), Output: nett.NewMatrixFromSlice([]float64{1})},
	// }

	// n := nett.New(&nett.TrainingConfig{
	// 	LearningRate: 0.4,
	// 	Epochs:       20,
	// 	Loss:         nett.EuclideanLoss,
	// })
	// n.AddLayers([]nett.Layer{
	// 	nett.NewFullyConnected(inputNodes, hiddenNodes, nett.Relu),
	// 	nett.NewFullyConnected(hiddenNodes, hiddenNodes, nett.Relu),
	// 	nett.NewFullyConnected(hiddenNodes, outputNodes, nett.Softmax),
	// })
	// n.Train(trainingData)

	// Let's test how well it did.
	// TODO (sno6): Implement
}
