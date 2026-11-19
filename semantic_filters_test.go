package imaging

import (
	"errors"
	"image"
	"image/color"
	"math"
	"testing"
)

func TestHistogramChecked(t *testing.T) {
	if _, err := HistogramChecked(nil); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil histogram: %v", err)
	}
	if _, err := HistogramChecked(nrgba(0, 0)); !errors.Is(err, ErrEmptyImage) {
		t.Fatalf("empty histogram: %v", err)
	}

	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.Pix = []uint8{
		0, 0, 0, 255, 100, 100, 100, 255,
		100, 100, 100, 255, 255, 255, 255, 255,
	}
	h, err := HistogramChecked(img)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, v := range h {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-12 {
		t.Fatalf("histogram not normalized: sum = %v", sum)
	}
	if h[0] != 0.25 {
		t.Fatalf("black bin = %v want 0.25", h[0])
	}
	if h[255] != 0.25 {
		t.Fatalf("white bin = %v want 0.25", h[255])
	}
	if h[100] != 0.5 {
		t.Fatalf("gray bin = %v want 0.5", h[100])
	}
}

func TestConvolveCheckedValidation(t *testing.T) {
	src := nrgba(4, 4)
	if _, err := ConvolveChecked(nil, make([]float64, 9), 1, nil); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil: %v", err)
	}
	if _, err := ConvolveChecked(src, make([]float64, 8), 1, nil); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("kernel length mismatch: %v", err)
	}
	if _, err := ConvolveChecked(src, make([]float64, 9), 2, nil); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("radius/length mismatch: %v", err)
	}
	// Radius above the hard global limit is rejected even on large images.
	bigRadius := MaxKernelRadius + 1
	bigKernel := make([]float64, (2*bigRadius+1)*(2*bigRadius+1))
	if _, err := ConvolveChecked(nrgba(4096, 4096), bigKernel, bigRadius, nil); !errors.Is(err, ErrRadiusTooLarge) {
		t.Fatalf("radius above MaxKernelRadius path: %v", err)
	}
	bad := make([]float64, 9)
	bad[4] = math.NaN()
	if _, err := ConvolveChecked(src, bad, 1, nil); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("NaN kernel: %v", err)
	}
	if _, err := ConvolveChecked(src, make([]float64, 9), -1, nil); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("negative radius: %v", err)
	}
}

func TestConvolveCheckedIdentityKernel(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*11 + 3) & 0xff)
	}
	kernel := []float64{
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,
	}
	got, err := ConvolveChecked(src, kernel, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Identity kernel must reproduce RGB exactly; alpha is copied.
	if !compareBytes(got.Pix, src.Pix, 0) {
		t.Fatal("identity convolution changed pixels")
	}

	// The caller's kernel slice must not be modified even with normalization.
	original := make([]float64, 9)
	copy(original, kernel)
	if _, err := ConvolveChecked(src, kernel, 1, &ConvolveOptions{Normalize: true}); err != nil {
		t.Fatal(err)
	}
	for i := range kernel {
		if kernel[i] != original[i] {
			t.Fatal("normalization mutated caller kernel slice")
		}
	}
}

func TestConvolveCheckedReferenceAndEdges(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			off := y*src.Stride + x*4
			src.Pix[off] = uint8((x + y*3) * 30)
			src.Pix[off+1] = src.Pix[off]
			src.Pix[off+2] = src.Pix[off]
			src.Pix[off+3] = 255
		}
	}
	kernel := []float64{
		1, 0, -1,
		1, 0, -1,
		1, 0, -1,
	}
	got, err := ConvolveChecked(src, kernel, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Independent reference with clamp-to-border sampling.
	want := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			var acc float64
			idx := 0
			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					ix := clampInt(x+kx, 0, 2)
					iy := clampInt(y+ky, 0, 2)
					acc += float64(src.Pix[iy*src.Stride+ix*4]) * kernel[idx]
					idx++
				}
			}
			off := y*want.Stride + x*4
			v := clamp(acc)
			want.Pix[off], want.Pix[off+1], want.Pix[off+2] = v, v, v
			want.Pix[off+3] = 255
		}
	}
	if !compareNRGBA(got, want, 0) {
		t.Fatalf("convolution reference mismatch\n got %v\nwant %v", got.Pix, want.Pix)
	}
}

func TestConvolveSinglePixel(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	src.Pix = []uint8{90, 90, 90, 255}
	kernel := []float64{
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,
	}
	got, err := Convolve3x3Checked(src, [9]float64(kernel), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !compareBytes(got.Pix, src.Pix, 0) {
		t.Fatalf("single-pixel convolution: got %v want %v", got.Pix, src.Pix)
	}
}

func TestBlurCheckedValidationAndTransparentBorder(t *testing.T) {
	if _, err := BlurChecked(nil, 1.0); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil blur: %v", err)
	}
	if _, err := BlurChecked(nrgba(4, 4), 0); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("zero sigma: %v", err)
	}
	if _, err := BlurChecked(nrgba(4, 4), -1); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("negative sigma: %v", err)
	}
	// An extreme sigma must be rejected before huge allocation.
	if _, err := BlurChecked(nrgba(64, 64), math.MaxFloat64); !errors.Is(err, ErrRadiusTooLarge) {
		t.Fatalf("max sigma: %v", err)
	}
	// Non-finite sigma is an invalid parameter rather than a radius issue.
	if _, err := BlurChecked(nrgba(64, 64), math.NaN()); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("NaN sigma: %v", err)
	}
	if _, err := BlurRadiusChecked(nrgba(4, 4), 4); !errors.Is(err, ErrRadiusTooLarge) {
		t.Fatalf("radius >= dimension: %v", err)
	}

	// Transparent border must not darken an opaque adjacent pixel.
	src := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	fillNRGBA(src, color.NRGBA{0, 0, 0, 0})
	for y := 0; y < 3; y++ {
		off := y*src.Stride + 4
		src.Pix[off], src.Pix[off+1], src.Pix[off+2], src.Pix[off+3] = 255, 255, 255, 255
	}
	got, err := BlurRadiusChecked(src, 1)
	if err != nil {
		t.Fatal(err)
	}
	center := got.Pix[1*got.Stride+4 : 1*got.Stride+8]
	if center[3] == 0 {
		t.Fatal("opaque center lost all alpha")
	}
	// Unpremultiplied center must stay bright (no dark fringe from black
	// transparent neighbors).
	if center[0] < 200 || center[1] < 200 || center[2] < 200 {
		t.Fatalf("dark edge from transparent neighbors: %v", center)
	}
}

func TestBlurDeterministicAcrossRuns(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 9, 7))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*19 + 7) & 0xff)
	}
	first, err := BlurChecked(src, 1.2)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 5; n++ {
		next, err := BlurChecked(src, 1.2)
		if err != nil {
			t.Fatal(err)
		}
		if !compareBytes(first.Pix, next.Pix, 0) {
			t.Fatal("blur output not reproducible across runs")
		}
	}
}

func TestDetectEdgesChecked(t *testing.T) {
	if _, err := DetectEdgesChecked(nil, EdgeSobel); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil edges: %v", err)
	}
	if _, err := DetectEdgesChecked(nrgba(4, 4), EdgeKernelKind(99)); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("unknown kernel: %v", err)
	}
	// Uniform image has zero gradient magnitude.
	uniform := nrgba(5, 5)
	fillNRGBA(uniform, color.NRGBA{120, 120, 120, 255})
	sobel, err := DetectEdgesChecked(uniform, EdgeSobel)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(sobel.Pix); i += 4 {
		if sobel.Pix[i] != 0 || sobel.Pix[i+1] != 0 || sobel.Pix[i+2] != 0 {
			t.Fatalf("uniform image produced nonzero edge: %v", sobel.Pix[i:i+4])
		}
		if sobel.Pix[i+3] != 255 {
			t.Fatal("edge detector changed alpha")
		}
	}
	lap, err := DetectEdgesChecked(uniform, EdgeLaplacian)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(lap.Pix); i += 4 {
		if lap.Pix[i] != 0 {
			t.Fatal("uniform image produced nonzero Laplacian response")
		}
	}

	// Single pixel is a valid edge input and must not error.
	one := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	one.Pix = []uint8{10, 20, 30, 255}
	if _, err := DetectEdgesChecked(one, EdgeSobel); err != nil {
		t.Fatalf("single pixel edge detection: %v", err)
	}
}

func TestFilterFailuresDoNotMutateSource(t *testing.T) {
	src := nrgba(4, 4)
	fillNRGBA(src, color.NRGBA{50, 100, 150, 255})
	before := snapshot(src)
	badInputs := []func() error{
		func() error { _, e := BlurChecked(src, math.MaxFloat64); return e },
		func() error {
			r := MaxKernelRadius + 1
			k := make([]float64, (2*r+1)*(2*r+1))
			_, e := ConvolveChecked(src, k, r, nil)
			return e
		},
		func() error { _, e := DetectEdgesChecked(src, EdgeKernelKind(99)); return e },
		func() error { _, e := HistogramChecked(nrgba(0, 0)); return e },
	}
	for i, fn := range badInputs {
		if err := fn(); err == nil {
			t.Fatalf("bad input %d expected error", i)
		}
		if !compareBytes(src.Pix, before, 0) {
			t.Fatalf("bad input %d mutated source", i)
		}
	}
}
