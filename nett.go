package nett

// A Layer...
type Layer interface {
	Init(p *Parameters, nextLayer Layer)
	Forward(p *Parameters) Matrix
	Backward(p *Parameters) Matrix
	Size() int
}

type Nett struct {
	cfg *TrainingConfig

	params *Parameters
	layers []Layer
}

type Parameters struct {
	weights []Matrix
	nodes   []Matrix
	deltas  []Matrix

	nLayers      int
	currentLayer int
	learningRate float64

	// Optimal output from training set.
	trainingOut Matrix
}

func (p *Parameters) LayerCount() int {
	return p.nLayers
}

func (p *Parameters) CurrentLayer() int {
	return p.currentLayer
}

func (p *Parameters) SetTrainingOutput(m Matrix) {
	p.trainingOut = m
}

func (p *Parameters) TrainingOutput() Matrix {
	return p.trainingOut
}

type (
	ActivationFunc func(n float64, deriv bool) float64
	LossFunc       func(netOut, realOut float64, deriv bool) float64
)

type TrainingConfig struct {
	LearningRate float64
	Epochs       int
	Loss         LossFunc
}

type TrainingSample struct {
	Input, Output Matrix
}

var DefaultConfig = &TrainingConfig{
	LearningRate: 0.4,
	Epochs:       100000,
	Loss:         EuclideanLoss,
}

func New(cfg *TrainingConfig, layers ...Layer) *Nett {
	if cfg == nil {
		cfg = DefaultConfig
	}
	n := &Nett{cfg: cfg}
	if len(layers) > 0 {
		n.InitModel(layers...)
	}
	return n
}

func (n *Nett) InitModel(layers ...Layer) {
	nLayers := len(layers)
	n.layers = layers
	n.params = &Parameters{
		weights:      make([]Matrix, nLayers-1),
		nodes:        make([]Matrix, nLayers),
		deltas:       make([]Matrix, nLayers),
		nLayers:      nLayers,
		learningRate: n.cfg.LearningRate,
	}
	for l := 0; l < nLayers-1; l++ {
		// TODO (sno6): Not too sure on this currentLayer biz, may be better to send through layer index to Layer methods.
		n.params.currentLayer = l
		layers[l].Init(n.params, layers[l+1])
	}
	n.params.currentLayer = 0
}

// Forward propogates an input matrix through the network.
func (n *Nett) Forward(input Matrix) Matrix {
	nLayers := len(n.layers)
	n.params.nodes[0] = input
	for l := 0; l < nLayers-1; l++ {
		n.params.currentLayer = l
		n.params.nodes[l+1] = n.layers[l].Forward(n.params)
	}
	return n.params.nodes[nLayers-1]
}

// Backward performs backpropogation in order to update the weights in the network.
func (n *Nett) Backward(om Matrix, tm Matrix) {
	n.params.SetTrainingOutput(tm)

	for l := len(n.layers) - 1; l > 0; l-- {
		n.params.currentLayer = l
		n.params.deltas[l] = n.layers[l].Backward(n.params)
	}
}

func (n *Nett) Train(data []*TrainingSample) {
	// TODO (sno6): Implement mini-batch.
	for i := 0; i < n.cfg.Epochs; i++ {
		for _, t := range data {
			n.Backward(n.Forward(t.Input), t.Output)
		}
	}
}

type FullyConnected struct {
	params *FullyConnectedParams
}

type FullyConnectedParams struct {
	Nodes      int
	Activation ActivationFunc
}

func NewFullyConnected(params *FullyConnectedParams) *FullyConnected {
	return &FullyConnected{params: params}
}

func (fc *FullyConnected) Init(params *Parameters, nextLayer Layer) {
	layer := params.CurrentLayer()
	params.weights[layer] = NewWeightMatrix(fc.params.Nodes, nextLayer.Size())
}

func (fc *FullyConnected) Forward(params *Parameters) Matrix {
	layer := params.CurrentLayer()
	return DotWithActivation(params.nodes[layer], params.weights[layer], fc.params.Activation)
}

func (fc *FullyConnected) Backward(params *Parameters) Matrix {
	currLayer := params.CurrentLayer()
	nLayers := params.LayerCount()

	// Handle the output layer deltas a little differently.
	if currLayer == nLayers-1 {
		return params.nodes[currLayer].SetForEachNew(func(o float64, x, y int) float64 {
			return fc.params.Activation(o, true) * (o - params.TrainingOutput().At(x, y)) // TODO (farleyschaefer): Get this from LossFunc..
		})
	}

	return params.nodes[currLayer].SetForEachNew(func(lOut float64, lX, lY int) float64 {
		weights := params.weights[currLayer].Row(lX)
		return params.deltas[currLayer+1].SetForEachNew(func(d float64, x, y int) float64 {
			w := params.weights[currLayer].At(x, lX)
			params.weights[currLayer].Set(x, lX, w-(params.learningRate*(lOut*d)))

			// Calculate new deltas for the current layer.
			return fc.params.Activation(lOut, true) * d * weights[x]
		}).Sum()
	})
}

func (fc *FullyConnected) Size() int {
	return fc.params.Nodes
}
