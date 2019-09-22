package nett

import (
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Matrix [][]float64

func NewMatrix(r, c int) Matrix {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}

func NewMatrixFromSlice(s []float64) Matrix {
	m := NewMatrix(1, len(s))
	for i, v := range s {
		m.Set(i, 0, float64(v))
	}
	return m
}

// Dims return the dimensions of a matrix (rows, columns).
func (m Matrix) Dims() (int, int) {
	return len(m), len(m[0])
}

// At returns the value in the matrix at [x, y].
func (m Matrix) At(x, y int) float64 {
	return m[y][x]
}

// Set sets the value at [x, y] to v.
func (m Matrix) Set(x, y int, v float64) {
	m[y][x] = v
}

// Row returns a row for a matrix.
func (m Matrix) Row(y int) []float64 {
	return m[y][0:]
}

// Flatten returns all elements of a matrix in a single slice.
func (m Matrix) Flatten() []float64 {
	r, c := m.Dims()
	flat := make([]float64, r*c)

	var i int
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			flat[i] = m.At(x, y)
			i++
		}
	}
	return flat
}

// SetForEach executes f on each value of the matrix and sets the value to the return value of f.
func (m Matrix) SetForEach(f func(val float64, x, y int) float64) {
	r, c := m.Dims()
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			v := m.At(x, y)
			m.Set(x, y, f(v, x, y))
		}
	}
}

// SetForEachNew is the same as ForEach except without overriding the original matrix.
func (m Matrix) SetForEachNew(f func(val float64, x, y int) float64) Matrix {
	r, c := m.Dims()
	n := NewMatrix(r, c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			v := m.At(x, y)
			n.Set(x, y, f(v, x, y))
		}
	}
	return n
}

// Dot performs the cross/dot product between two matrices.
func Dot(a, b Matrix) Matrix {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ac != br {
		panic("Dot: Dimension mismatch")
	}
	m := NewMatrix(ar, bc)
	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			var val float64
			for k := 0; k < br; k++ {
				val += b[k][j] * a[i][k]
			}
			m.Set(j, i, val)
		}
	}
	return m
}

// Sum sums a matrix into a single value.
func (m Matrix) Sum() float64 {
	var sum float64
	r, c := m.Dims()
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			sum += m.At(x, y)
		}
	}
	return sum
}

func (m Matrix) T() Matrix {
	x, y := m.Dims()
	out := NewMatrix(y, x)
	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			out.Set(i, j, m.At(j, i))
		}
	}
	return out
}

func Sigmoid(n float64, deriv bool) float64 {
	if deriv {
		return n * (1 - n)
	}
	return 1 / (1 + math.Exp(-n))
}

func Softmax(n float64, deriv bool) float64 {
	// TODO (sno6): Implement this goi.
	return n
}

func DotWithActivation(a, b Matrix, f ActivationFunc) Matrix {
	out := Dot(a, b)
	out.SetForEach(func(v float64, x, y int) float64 {
		return f(v, false)
	})
	return out
}

func NewWeightMatrix(r, c int) Matrix {
	m := NewMatrix(r, c)
	m.SetForEach(func(v float64, x, y int) float64 {
		return rand.Float64()
	})
	return m
}

func EuclideanLoss(t float64, o float64, deriv bool) float64 {
	if deriv {
		return o - t
	}
	return (1.0 / 2.0) * math.Pow(t-o, 2)
}

func (m Matrix) Pretty() string {
	var b strings.Builder

	r, c := m.Dims()
	for y := 0; y < r; y++ {
		b.WriteString("[ ")
		for x := 0; x < c; x++ {
			v := m.At(x, y)
			b.WriteString(strconv.FormatFloat(v, 'f', 4, 64) + " ")
		}
		b.WriteString(" ]\n")
	}
	return b.String()
}
