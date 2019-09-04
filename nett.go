package nett

type Nett struct {
	cfg                    *Config
	weights, nodes, deltas []Matrix
}

type Config struct {
	LearningRate                float64
	TrainingEpochs              int
	InputN, OutputN             int
	HiddenLayersN, HiddenNodesN int
}

type TrainingSample struct {
	Input, Output Matrix
}

var DefaultConfig = &Config{
	LearningRate:   0.4,
	TrainingEpochs: 100000,
	InputN:         2,
	OutputN:        1,
	HiddenLayersN:  2,
	HiddenNodesN:   10,
}

func New(cfg *Config) *Nett {
	if cfg == nil {
		cfg = DefaultConfig
	}
	n := &Nett{cfg: cfg}
	n.InitWeights()
	n.InitNodes()
	n.InitDeltas()
	return n
}

func (n *Nett) InitNodes() {
	if n.nodes == nil {
		n.nodes = make([]Matrix, n.Layers())
	}
}

func (n *Nett) InitDeltas() {
	if n.deltas == nil {
		n.deltas = make([]Matrix, n.Layers())
	}
}

func (n *Nett) InitWeights() {
	if n.weights == nil {
		n.weights = make([]Matrix, n.Layers()-1)
	}

	// Set input & output weight dimensions.
	inp := 0
	out := len(n.weights) - 1
	n.weights[inp] = NewWeightMatrix(n.cfg.InputN, n.cfg.HiddenNodesN)
	n.weights[out] = NewWeightMatrix(n.cfg.HiddenNodesN, n.cfg.OutputN)

	// Set hidden layer weights.
	for i := inp + 1; i < out; i++ {
		n.weights[i] = NewWeightMatrix(n.cfg.HiddenNodesN, n.cfg.HiddenNodesN)
	}
}

func (n *Nett) Layers() int {
	return 1 + n.cfg.HiddenLayersN + 1
}

// Forward propogates an input matrix through the network.
func (n *Nett) Forward(input Matrix) Matrix {
	layers := n.Layers()
	n.nodes[0] = input
	for l := 1; l < layers; l++ {
		n.nodes[l] = DotWithSigmoid(n.nodes[l-1], n.weights[l-1])
	}
	return n.nodes[layers-1]
}

// Backward performs backpropogation in order to update the weights in the network.
func (n *Nett) Backward(om Matrix, tm Matrix) {
	nLayers := n.Layers()

	// Calculate deltas for error layer.
	n.deltas[nLayers-1] = om.SetForEachNew(func(o float64, x, y int) float64 {
		return SigmoidDeriv(o) * (o - tm.At(x, y))
	})

	// Calculate the deltas & adjust weights for each hidden layer.
	for l := nLayers - 2; l >= 0; l-- {
		n.deltas[l] = n.nodes[l].SetForEachNew(func(lOut float64, lX, lY int) float64 {
			weights := n.weights[l].Row(lX)
			return n.deltas[l+1].SetForEachNew(func(d float64, x, y int) float64 {
				// Update weights in the current layer.
				w := n.weights[l].At(x, lX)
				n.weights[l].Set(x, lX, w-(n.cfg.LearningRate*(lOut*d)))

				// Calculate new deltas for the current layer.
				return SigmoidDeriv(lOut) * d * weights[x]
			}).Sum()
		})
	}
}

// TODO (sno6): Super simple, this could do with improving..
func (n *Nett) Train(data []*TrainingSample) {
	for i := 0; i < n.cfg.TrainingEpochs; i++ {
		for _, t := range data {
			n.Backward(n.Forward(t.Input), t.Output)
		}
	}
}
