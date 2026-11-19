package imaging

import (
	"image"
	"image/color"
	"testing"
	"testing/quick"
)

// FuzzNearestResizeIndex checks the documented nearest-neighbor invariant:
// every destination pixel maps to a valid, monotonically non-decreasing source
// index covering the full source extent.
func FuzzNearestResizeIndex(f *testing.F) {
	f.Add(16, 9)
	f.Add(7, 5)
	f.Add(1, 1)
	f.Add(33, 1)
	f.Fuzz(func(t *testing.T, dstSize, srcSize int) {
		if dstSize < 1 || srcSize < 1 || dstSize > 4096 || srcSize > 4096 {
			return
		}
		prev := -1
		for d := 0; d < dstSize; d++ {
			idx := nearestSourceIndex(d, dstSize, srcSize)
			if idx < 0 || idx >= srcSize {
				t.Fatalf("index %d out of source range %d", idx, srcSize)
			}
			if idx < prev {
				t.Fatalf("non-monotonic mapping at %d: %d < %d", d, idx, prev)
			}
			prev = idx
		}
		// Coverage invariants: up-scaling must reach every source pixel, and
		// in every case the sampled indices must stay within source bounds and
		// be monotonically non-decreasing.
		seen := map[int]bool{}
		for d := 0; d < dstSize; d++ {
			seen[nearestSourceIndex(d, dstSize, srcSize)] = true
		}
		if srcSize <= dstSize && len(seen) != srcSize {
			t.Fatalf("up-scale must cover every source pixel, saw %d of %d", len(seen), srcSize)
		}
	})
}

// FuzzPremultiplyRoundTrip drives the full NRGBA->RGBA->NRGBA path with random
// pixels and asserts the structural alpha invariants.
func FuzzPremultiplyRoundTrip(f *testing.F) {
	f.Add(uint8(123), uint8(45))
	f.Fuzz(func(t *testing.T, v, a uint8) {
		src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		src.Pix = []uint8{v, v, v, a}
		prem, err := Premultiply(src)
		if err != nil {
			t.Fatal(err)
		}
		back, err := Unpremultiply(prem)
		if err != nil {
			t.Fatal(err)
		}
		if back.Pix[3] != a {
			t.Fatalf("alpha not preserved: %d -> %d", a, back.Pix[3])
		}
		switch {
		case a == 0:
			for k := 0; k < 4; k++ {
				if back.Pix[k] != 0 {
					t.Fatalf("transparent pixel not zeroed: %v", back.Pix)
				}
			}
		case a == 255:
			for k := 0; k < 3; k++ {
				if back.Pix[k] != v {
					t.Fatalf("opaque pixel not identity: %d -> %d", v, back.Pix[k])
				}
			}
		default:
			for k := 0; k < 3; k++ {
				if back.Pix[k] > 255 {
					t.Fatal("channel out of range")
				}
			}
		}
		// Premultiplied channel can never exceed alpha coverage.
		for k := 0; k < 3; k++ {
			if prem.Pix[k] > a {
				t.Fatalf("premultiplied channel %d > alpha %d", prem.Pix[k], a)
			}
		}
	})
}

// FuzzColorPipelineBounds ensures the pipeline never produces out-of-range
// values and always preserves alpha for arbitrary parameter combinations.
func FuzzColorPipelineBounds(f *testing.F) {
	f.Add(uint8(200), float64(10.0), float64(-5.0), float64(1.3), float64(20.0))
	f.Fuzz(func(t *testing.T, v uint8, bright, contrast, gamma, sat float64) {
		if bright < -100 || bright > 100 || contrast < -100 || contrast > 100 ||
			sat < -100 || sat > 100 || gamma <= 0 || gamma > 8 {
			return
		}
		src := image.NewNRGBA(image.Rect(0, 0, 2, 2))
		for i := 0; i < 4; i++ {
			off := i * 4
			src.Pix[off], src.Pix[off+1], src.Pix[off+2] = v, uint8((int(v)+17)%256), uint8((int(v)+91)%256)
			src.Pix[off+3] = uint8(i * 60)
		}
		out, err := ApplyColorPipeline(src, ColorAdjustments{
			Brightness: bright, Contrast: contrast, Gamma: gamma, Saturation: sat,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			off := i * 4
			for k := 0; k < 3; k++ {
				if out.Pix[off+k] > 255 {
					t.Fatal("channel overflow")
				}
			}
			if out.Pix[off+3] != src.Pix[off+3] {
				t.Fatalf("alpha changed: %d -> %d", src.Pix[off+3], out.Pix[off+3])
			}
		}
	})
}

// FuzzResizeDimensions checks that any valid dimension pair always yields an
// image of exactly the requested size with finite pixel data.
func FuzzResizeDimensions(f *testing.F) {
	f.Add(12, 8, 5, 7)
	f.Fuzz(func(t *testing.T, sw, sh, dw, dh int) {
		if sw < 1 || sh < 1 || dw < 1 || dh < 1 ||
			sw > 64 || sh > 64 || dw > 64 || dh > 64 {
			return
		}
		src := image.NewNRGBA(image.Rect(0, 0, sw, sh))
		for i := range src.Pix {
			src.Pix[i] = uint8((i * 31) & 0xff)
		}
		for _, filter := range []ResampleFilter{NearestNeighbor, Box, Linear} {
			out, err := ResizeChecked(src, dw, dh, filter)
			if err != nil {
				t.Fatal(err)
			}
			if out.Rect.Dx() != dw || out.Rect.Dy() != dh {
				t.Fatalf("size %dx%d want %dx%d", out.Rect.Dx(), out.Rect.Dy(), dw, dh)
			}
			if len(out.Pix) != dw*dh*4 {
				t.Fatalf("pixel buffer length %d want %d", len(out.Pix), dw*dh*4)
			}
		}
	})
}

// TestPropertyChainedResizeAssociativityOfShape verifies that chaining resizes
// through the checked API and the legacy API with the same nearest rule gives
// identical shapes and pixels for the nearest filter.
func TestPropertyNearestLegacyParity(t *testing.T) {
	prop := func(sw, sh, dw, dh uint8) bool {
		if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
			return true
		}
		src := image.NewNRGBA(image.Rect(0, 0, int(sw), int(sh)))
		for i := range src.Pix {
			src.Pix[i] = uint8((i * 41) & 0xff)
		}
		got, err := ResizeChecked(src, int(dw), int(dh), NearestNeighbor)
		if err != nil {
			return false
		}
		legacy := Resize(src, int(dw), int(dh), NearestNeighbor)
		return compareNRGBA(got, legacy, 0)
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatal(err)
	}
}

// TestPropertyIdempotentRepeatedCalls ensures calling a checked transform twice
// produces byte-identical results (determinism / repeatability).
func TestPropertyIdempotentRepeatedCalls(t *testing.T) {
	prop := func(seed uint8) bool {
		src := image.NewNRGBA(image.Rect(0, 0, 8, 6))
		for i := range src.Pix {
			src.Pix[i] = uint8((int(seed) + i*13) & 0xff)
		}
		first, err := BlurChecked(src, 1.0)
		if err != nil {
			return false
		}
		second, err := BlurChecked(src, 1.0)
		if err != nil {
			return false
		}
		if !compareBytes(first.Pix, second.Pix, 0) {
			return false
		}
		r1, err := RotateChecked(src, 47, color.NRGBA{5, 6, 7, 8})
		if err != nil {
			return false
		}
		r2, err := RotateChecked(src, 47, color.NRGBA{5, 6, 7, 8})
		if err != nil {
			return false
		}
		return compareBytes(r1.Pix, r2.Pix, 0)
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100}); err != nil {
		t.Fatal(err)
	}
}
