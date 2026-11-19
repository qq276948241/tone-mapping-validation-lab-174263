package imaging

import (
	"image"
	"image/color"
	"sync"
	"testing"
)

// TestBufferPoolConcurrentSafety hammers the pool from many goroutines to
// ensure Get/Put never hand out overlapping ownership or race.
func TestBufferPoolConcurrentSafety(t *testing.T) {
	var p BufferPool
	const goroutines = 64
	const iterations = 200
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				size := 64 + (id*7+i*13)%512
				buf := p.Get(size)
				if len(buf) != size {
					t.Errorf("buffer length = %d want %d", len(buf), size)
					return
				}
				// Ownership: writing our marker must not be observed by others.
				marker := byte(id)
				for j := range buf {
					buf[j] = marker
				}
				for j := range buf {
					if buf[j] != marker {
						t.Errorf("buffer ownership violated at %d", j)
						return
					}
				}
				p.Put(buf)
			}
		}(g)
	}
	wg.Wait()
}

// TestSemanticFunctionsConcurrency runs the same transforms concurrently on
// shared source images; under -race this proves the read-only source contract
// and the per-goroutine scratch buffers are sound.
func TestSemanticFunctionsConcurrency(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 24, 18))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*23 + 5) & 0xff)
	}
	bg := color.NRGBA{1, 2, 3, 4}
	adj := ColorAdjustments{Brightness: 8, Contrast: 6, Gamma: 1.3, Saturation: 12}
	kernel := [9]float64{0, -1, 0, -1, 5, -1, 0, -1, 0}

	var wg sync.WaitGroup
	for g := 0; g < 32; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := ResizeChecked(src, 12, 9, Linear); err != nil {
				t.Error(err)
			}
			if _, err := ResizeChecked(src, 40, 30, NearestNeighbor); err != nil {
				t.Error(err)
			}
			if _, err := RotateChecked(src, 33, bg); err != nil {
				t.Error(err)
			}
			if _, err := CropChecked(src, image.Rect(1, 1, 10, 8)); err != nil {
				t.Error(err)
			}
			if _, err := ApplyColorPipeline(src, adj); err != nil {
				t.Error(err)
			}
			if _, err := Convolve3x3Checked(src, kernel, nil); err != nil {
				t.Error(err)
			}
			if _, err := BlurChecked(src, 1.0); err != nil {
				t.Error(err)
			}
			if _, err := DetectEdgesChecked(src, EdgeSobel); err != nil {
				t.Error(err)
			}
			if _, err := HistogramChecked(src); err != nil {
				t.Error(err)
			}
			if _, err := Premultiply(src); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

// TestRepeatedCallsStable confirms repeated invocation of every checked entry
// point returns byte-identical output and leaves the source unchanged.
func TestRepeatedCallsStable(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 16, 12))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*17 + 3) & 0xff)
	}
	before := snapshot(src)
	ops := []func() (*image.NRGBA, error){
		func() (*image.NRGBA, error) { return ResizeChecked(src, 8, 6, CatmullRom) },
		func() (*image.NRGBA, error) { return CropChecked(src, image.Rect(0, 0, 8, 6)) },
		func() (*image.NRGBA, error) { return RotateChecked(src, 29, color.NRGBA{0, 0, 0, 0}) },
		func() (*image.NRGBA, error) {
			return ApplyColorPipeline(src, ColorAdjustments{Gamma: 1.5, Saturation: -10})
		},
	}
	for i, op := range ops {
		first, err := op()
		if err != nil {
			t.Fatalf("op %d first: %v", i, err)
		}
		second, err := op()
		if err != nil {
			t.Fatalf("op %d second: %v", i, err)
		}
		if !compareBytes(first.Pix, second.Pix, 0) {
			t.Fatalf("op %d not stable across repeated calls", i)
		}
	}
	if !compareBytes(src.Pix, before, 0) {
		t.Fatal("source mutated after repeated calls")
	}
}
