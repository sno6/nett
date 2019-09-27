package main

const (
// inputNodes  = 784
// hiddenNodes = 512
// outputNodes = 10

// imagesLoc        = "./data/train-images-idx3-ubyte"
// labelsLoc        = "./data/train-labels-idx1-ubyte"
// testingLabelsLoc = "./data/t10k-labels-idx1-ubyte"
)

func main() {
	// r, err := mnist.New(&mnist.Files{
	// 	TrainingImagesLoc: imagesLoc,
	// 	TrainingLabelsLoc: labelsLoc,
	// 	TestingLabelsLoc:  testingLabelsLoc,
	// })
	// if err != nil {
	// 	log.Fatalf("mnist: error initialising: %v", err)
	// }

	// n := nett.New(&nett.TrainingConfig{
	// 	LearningRate: 0.4,
	// 	Epochs:       20,
	// 	LossFunc:     nett.EuclideanLoss,
	// })

	// // Train a simple MLP MNIST model, 20 epochs, 784 -> 512 (RELU) -> 512 (RELU) -> 10 (Softmax)
	// n.InitModel([]nett.Layer{
	// 	nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: inputNodes, Activation: nett.Relu}),
	// 	nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: hiddenNodes, Activation: nett.Relu}),
	// 	nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: hiddenNodes, Activation: nett.Relu}),
	// 	nett.NewFullyConnected(&nett.FullyConnectedParams{Nodes: outputNodes, Activation: nett.Softmax}),
	// }...)

	// // data := getTrainingData(r)
	// // n.Train(data)

	// // Test the network on some random images.
	// for i := 0; i < 5; i++ {
	// 	offset := i + 1
	// 	img := newMatrix(r.TestingImages.GetFlatImage(offset)...)
	// 	lab := newMatrixForLabel(int(r.TestingLabels.GetLabel(offset)))

	// 	out := n.Forward(img)
	// 	fmt.Printf("Passing test image with label %d through the network\n", lab)
	// 	fmt.Println(out.Pretty())
	// }
}

// func getTrainingData(r *mnist.Reader) []*nett.TrainingSample {
// 	samples := make([]*nett.TrainingSample, r.TrainingImages.Count)
// 	for i := range samples {
// 		samples[i] = &nett.TrainingSample{
// 			Input:  newMatrix(r.TrainingImages.GetFlatImage(i)...),
// 			Output: newMatrixForLabel(int(r.TrainingLabels.GetLabel(i))),
// 		}
// 	}
// 	return samples
// }

// func newMatrix(vals ...uint8) nett.Matrix {
// 	m := nett.NewMatrix(1, len(vals))
// 	for i, v := range vals {
// 		m.Set(i, 0, float64(v))
// 	}
// 	return m
// }

// func newMatrixForLabel(n int) nett.Matrix {
// 	s := make([]float64, 10)
// 	s[n] = 1
// 	return nett.NewMatrixFromSlice(s)
// }
