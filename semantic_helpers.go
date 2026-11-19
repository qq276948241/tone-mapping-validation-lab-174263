package imaging

import (
	"context"
	"image"
	"image/color"
	"image/color/palette"
	"io"
	"math"
	"runtime"
	"sync/atomic"

	"golang.org/x/image/bmp"
	"golang.org/x/image/tiff"
)

// nanValue and powFloat centralize math helpers used across production code
// and the semantic test suite so both sides share the exact same operations.
var nanValue = math.NaN()

func powFloat(x, e float64) float64 { return math.Pow(x, e) }

// opaqueBlack is a reusable background color for benchmark rotations.
func opaqueBlack() color.Color { return color.NRGBA{0, 0, 0, 255} }

// batchBenchContext returns the background context used by benchmarks.
func batchBenchContext() context.Context { return context.Background() }

// Thin wrappers around the extended-format encoders so that semantic_io.go
// stays free of format-specific import plumbing.

func tiffEncode(w io.Writer, img image.Image) error {
	return tiff.Encode(w, img, &tiff.Options{Compression: tiff.Deflate, Predictor: true})
}

func bmpEncode(w io.Writer, img image.Image) error {
	return bmp.Encode(w, img)
}

// colorPalette is the palette type used by GIF encoding.
type colorPalette = color.Palette

func plan9Palette(n int) color.Palette {
	full := palette.Plan9
	if n >= len(full) {
		return full
	}
	return full[:n]
}

func atomicLoadMaxProcs() int64 {
	return atomic.LoadInt64(&maxProcs)
}

func gomaxprocs() int {
	return runtime.GOMAXPROCS(0)
}
