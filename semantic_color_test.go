package imaging

import (
	"errors"
	"image"
	"image/color"
	"testing"
)

func TestMul8Div8Rounding(t *testing.T) {
	// mul8 is round-half-up integer scaling and must be exact at the extremes.
	if got := mul8(255, 255); got != 255 {
		t.Fatalf("mul8(255,255) = %d", got)
	}
	if got := mul8(0, 128); got != 0 {
		t.Fatalf("mul8(0,128) = %d", got)
	}
	if got := mul8(255, 0); got != 0 {
		t.Fatalf("mul8(255,0) = %d", got)
	}
	// Explicit half-up check: 1*128+127 over 255 -> (255)/255 = 1.
	if got := mul8(1, 128); got != 1 {
		t.Fatalf("mul8(1,128) = %d want 1", got)
	}
	// Round-trip error is bounded by the premultiplied grid spacing plus the
	// rounding half-step on each conversion. At high alpha the spacing shrinks
	// below one unit so the result is exact within a single quantization step.
	for a := 1; a < 256; a++ {
		got := div8(mul8(120, uint8(a)), uint8(a))
		bound := 256/a + 1
		diff := int(got) - 120
		if diff < 0 {
			diff = -diff
		}
		if diff > bound {
			t.Fatalf("premultiply round trip error %d > bound %d at alpha %d", diff, bound, a)
		}
	}
}

func TestPremultiplyUnpremultiplyRoundTrip(t *testing.T) {
	// Exhaustive structural check over representative (value, alpha) pairs.
	values := []int{0, 1, 2, 16, 127, 128, 200, 254, 255}
	type sample struct{ v, a int }
	var samples []sample
	for a := 0; a < 256; a++ {
		for _, v := range values {
			samples = append(samples, sample{v, a})
		}
	}
	src := image.NewNRGBA(image.Rect(0, 0, len(samples), 1))
	for i, s := range samples {
		off := i * 4
		src.Pix[off], src.Pix[off+1], src.Pix[off+2], src.Pix[off+3] =
			uint8(s.v), uint8(s.v), uint8(s.v), uint8(s.a)
	}
	prem, err := Premultiply(src)
	if err != nil {
		t.Fatal(err)
	}
	back, err := Unpremultiply(prem)
	if err != nil {
		t.Fatal(err)
	}

	for i, s := range samples {
		v, a := s.v, s.a
		off := i * 4
		switch {
		case a == 0:
			// Fully transparent: every channel including alpha must be zero.
			for k := 0; k < 4; k++ {
				if back.Pix[off+k] != 0 {
					t.Fatalf("alpha 0 pixel not zeroed: a=%d v=%d pix=%v", a, v, back.Pix[off:off+4])
				}
			}
		case a == 255:
			for k := 0; k < 3; k++ {
				if back.Pix[off+k] != uint8(v) {
					t.Fatalf("alpha 255 not identity: got %d want %d", back.Pix[off+k], v)
				}
			}
			if back.Pix[off+3] != 255 {
				t.Fatal("alpha 255 not preserved")
			}
		default:
			if back.Pix[off+3] != uint8(a) {
				t.Fatalf("alpha changed: %d -> %d", a, back.Pix[off+3])
			}
			premCoverage := mul8(uint8(v), uint8(a))
			for k := 0; k < 3; k++ {
				got := int(back.Pix[off+k])
				if premCoverage == 0 {
					if got != 0 {
						t.Fatalf("zero coverage unpremultiplied to %d at a=%d v=%d", got, a, v)
					}
					continue
				}
				// Total round-trip error against the ORIGINAL value is bounded
				// by one coverage-grid spacing (255/a) plus a rounding step:
				// premultiplication rounds onto the grid, unpremultiplication
				// rounds half-up on the way back.
				if a >= 16 {
					allowed := 255/a + 1
					if d := absint(got - v); d > allowed {
						t.Fatalf("round trip error %d > allowed %d (a=%d v=%d got=%d)",
							d, allowed, a, v, got)
					}
				}
				if got < 0 || got > 255 {
					t.Fatalf("channel out of range: %d", got)
				}
			}
		}
	}
}

func TestTransparentPixelsHaveNoColorBleed(t *testing.T) {
	// A transparent region carrying bright RGB must premultiply to black so
	// it cannot contaminate visible edges.
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.Pix = []uint8{255, 255, 255, 0, 10, 20, 30, 255}
	prem, err := Premultiply(src)
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 4; k++ {
		if prem.Pix[k] != 0 {
			t.Fatalf("transparent pixel premultiplied to non-zero channel %d: %d", k, prem.Pix[k])
		}
	}
	back, err := Unpremultiply(prem)
	if err != nil {
		t.Fatal(err)
	}
	if back.Pix[3] != 0 || back.Pix[0] != 0 || back.Pix[1] != 0 || back.Pix[2] != 0 {
		t.Fatalf("unpremultiplied transparent pixel: %v", back.Pix[:4])
	}
}

func TestCompositeOverNoDarkEdge(t *testing.T) {
	// Semi-transparent red over white must stay bright, never turn dark.
	bg := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	bg.Pix = []uint8{255, 255, 255, 255}
	fg := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	fg.Pix = []uint8{255, 0, 0, 128}
	out, err := CompositeOver(bg, fg)
	if err != nil {
		t.Fatal(err)
	}
	if out.Pix[0] < 240 {
		t.Fatalf("dark edge on red-over-white R channel: %d", out.Pix[0])
	}
	if out.Pix[1] > 140 || out.Pix[2] > 140 {
		t.Fatalf("green/blue should be damped, got %v", out.Pix[:4])
	}

	// Fully transparent foreground leaves the background exactly untouched.
	clear := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	clear.Pix = []uint8{0, 0, 0, 0}
	out, err = CompositeOver(bg, clear)
	if err != nil {
		t.Fatal(err)
	}
	if !compareBytes(out.Pix, bg.Pix, 0) {
		t.Fatal("transparent foreground altered background")
	}

	// Fully opaque foreground replaces background.
	opaque := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	opaque.Pix = []uint8{7, 8, 9, 255}
	out, err = CompositeOver(bg, opaque)
	if err != nil {
		t.Fatal(err)
	}
	if !compareBytes(out.Pix, opaque.Pix, 0) {
		t.Fatal("opaque foreground did not replace background")
	}

	if _, err := CompositeOver(nil, fg); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil background: %v", err)
	}
	mismatch := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	if _, err := CompositeOver(bg, mismatch); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: %v", err)
	}
}

func TestColorPipelineValidationAndOrder(t *testing.T) {
	src := nrgba(2, 2)
	fillNRGBA(src, color.NRGBA{100, 150, 200, 255})

	bad := []ColorAdjustments{
		{Brightness: 101},
		{Brightness: -101},
		{Contrast: 200},
		{Saturation: -150},
		{Gamma: -1},
		{Gamma: mathNaN()},
	}
	for i, adj := range bad {
		if _, err := ApplyColorPipeline(src, adj); !errors.Is(err, ErrInvalidParameter) {
			t.Fatalf("bad adjustment %d: got %v want ErrInvalidParameter", i, err)
		}
	}
	if _, err := ApplyColorPipeline(nil, ColorAdjustments{}); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil pipeline input: %v", err)
	}

	// Identity pipeline must return identical pixels.
	id, err := ApplyColorPipeline(src, ColorAdjustments{Gamma: 1.0})
	if err != nil {
		t.Fatal(err)
	}
	if !compareBytes(id.Pix, src.Pix, 0) {
		t.Fatal("identity pipeline changed pixels")
	}
}

func TestColorPipelineFixedStagesReference(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	src.Pix = []uint8{60, 120, 200, 255}
	adj := ColorAdjustments{Brightness: 10, Contrast: 15, Gamma: 1.4, Saturation: -20}

	got, err := ApplyColorPipeline(src, adj)
	if err != nil {
		t.Fatal(err)
	}

	// Explicit reference implementing the documented order and the single
	// round-half-up truncation point.
	r, g, b := 60.0, 120.0, 200.0
	r += 255.0 * 10 / 100
	g += 255.0 * 10 / 100
	b += 255.0 * 10 / 100
	v := 1.15
	cf := 1 / (2 - v)
	r = 127.5 + (r-127.5)*cf
	g = 127.5 + (g-127.5)*cf
	b = 127.5 + (b-127.5)*cf
	stage := func(x float64) uint8 {
		if x < 0 {
			x = 0
		}
		if x > 255 {
			x = 255
		}
		return roundToUint8(mathPowRef(x/255, 1/1.4) * 255)
	}
	r8, g8, b8 := stage(r), stage(g), stage(b)
	h, s, l := rgbToHSL(r8, g8, b8)
	s *= 0.8
	wantR, wantG, wantB := hslToRGB(h, s, l)

	want := []uint8{wantR, wantG, wantB, 255}
	if !compareBytes(got.Pix, want, 0) {
		t.Fatalf("pipeline output %v does not match ordered reference %v", got.Pix, want)
	}
}

func TestColorPipelineStageOrderMatters(t *testing.T) {
	// Reusing the same parameters in two different orderings must produce
	// different results, proving the documented order is actually enforced.
	src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	src.Pix = []uint8{40, 90, 230, 255}
	adj := ColorAdjustments{Brightness: 25, Contrast: 30, Gamma: 0.8}
	got, err := ApplyColorPipeline(src, adj)
	if err != nil {
		t.Fatal(err)
	}

	// Manual alternative order: gamma, contrast, brightness.
	r, g, b := 40.0, 90.0, 230.0
	gammaStep := func(x float64) float64 {
		v := mathPowRef(x/255, 1/0.8) * 255
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return v
	}
	r, g, b = gammaStep(r), gammaStep(g), gammaStep(b)
	cf := 1 / (2 - 1.3)
	r = 127.5 + (r-127.5)*cf
	g = 127.5 + (g-127.5)*cf
	b = 127.5 + (b-127.5)*cf
	r += 255 * 0.25
	g += 255 * 0.25
	b += 255 * 0.25
	alt := []uint8{roundToUint8(r), roundToUint8(g), roundToUint8(b), 255}
	if compareBytes(got.Pix, alt, 0) {
		t.Fatal("pipeline is insensitive to stage ordering; order is not enforced")
	}
}

func TestRoundToUint8HalfUp(t *testing.T) {
	cases := []struct {
		in   float64
		want uint8
	}{
		{-1, 0}, {0, 0}, {0.49, 0}, {0.5, 1}, {1.5, 2},
		{254.5, 255}, {255, 255}, {300, 255},
	}
	for _, tc := range cases {
		if got := roundToUint8(tc.in); got != tc.want {
			t.Fatalf("roundToUint8(%v) = %d want %d", tc.in, got, tc.want)
		}
	}
}

func TestIndividualCheckedAdjustments(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	src.Pix = []uint8{128, 128, 128, 255}

	b, err := BrightnessChecked(src, 10)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := ApplyColorPipeline(src, ColorAdjustments{Brightness: 10})
	if !compareBytes(b.Pix, ref.Pix, 0) {
		t.Fatal("BrightnessChecked disagrees with pipeline")
	}
	c, err := ContrastChecked(src, -10)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ = ApplyColorPipeline(src, ColorAdjustments{Contrast: -10})
	if !compareBytes(c.Pix, ref.Pix, 0) {
		t.Fatal("ContrastChecked disagrees with pipeline")
	}
	g, err := GammaChecked(src, 2.2)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ = ApplyColorPipeline(src, ColorAdjustments{Gamma: 2.2})
	if !compareBytes(g.Pix, ref.Pix, 0) {
		t.Fatal("GammaChecked disagrees with pipeline")
	}
	s, err := SaturationChecked(src, 30)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ = ApplyColorPipeline(src, ColorAdjustments{Saturation: 30})
	if !compareBytes(s.Pix, ref.Pix, 0) {
		t.Fatal("SaturationChecked disagrees with pipeline")
	}
	if _, err := GammaChecked(src, -1.0); !errors.Is(err, ErrInvalidParameter) {
		t.Fatal("negative gamma must be rejected")
	}
	if _, err := GammaChecked(src, mathNaN()); !errors.Is(err, ErrInvalidParameter) {
		t.Fatal("NaN gamma must be rejected")
	}
}

func TestAlphaUntouchedByColorPipeline(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	src.Pix = []uint8{
		10, 20, 30, 0,
		40, 50, 60, 128,
		70, 80, 90, 255,
	}
	out, err := ApplyColorPipeline(src, ColorAdjustments{Brightness: 15, Contrast: 10, Gamma: 1.6, Saturation: 25})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if out.Pix[i*4+3] != src.Pix[i*4+3] {
			t.Fatalf("alpha changed at pixel %d: %d -> %d", i, src.Pix[i*4+3], out.Pix[i*4+3])
		}
	}
}

func mathNaN() float64 { return nanValue }

func mathPowRef(x, e float64) float64 { return powFloat(x, e) }
