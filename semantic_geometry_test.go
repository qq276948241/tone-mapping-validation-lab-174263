package imaging

import (
	"errors"
	"image"
	"image/color"
	"math"
	"testing"
)

func nrgba(w, h int) *image.NRGBA {
	return image.NewNRGBA(image.Rect(0, 0, w, h))
}

func fillNRGBA(img *image.NRGBA, c color.NRGBA) {
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = c.R, c.G, c.B, c.A
	}
}

func snapshot(img *image.NRGBA) []uint8 {
	out := make([]uint8, len(img.Pix))
	copy(out, img.Pix)
	return out
}

func TestValidateImageErrors(t *testing.T) {
	if _, err := validateImage(nil); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil image: got %v want ErrNilImage", err)
	}
	var typedNil *image.NRGBA
	if _, err := validateImage(typedNil); !errors.Is(err, ErrNilImage) {
		t.Fatalf("typed nil image: got %v want ErrNilImage", err)
	}
	empty := &image.NRGBA{}
	if _, err := validateImage(empty); !errors.Is(err, ErrEmptyImage) {
		t.Fatalf("empty image: got %v want ErrEmptyImage", err)
	}
	if err := validateSize(0, 5); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("zero width: got %v want ErrInvalidSize", err)
	}
	if err := validateSize(5, -1); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("negative height: got %v want ErrInvalidSize", err)
	}
	if err := validateSize(MaxImageDimension+1, 5); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("oversized: got %v want ErrInvalidSize", err)
	}
	if err := validateSize(4, 4); err != nil {
		t.Fatalf("valid size rejected: %v", err)
	}
}

func TestValidateRadiusErrors(t *testing.T) {
	cases := []struct {
		radius, w, h int
		want         error
	}{
		{-1, 10, 10, ErrInvalidParameter},
		{MaxKernelRadius + 1, 4096, 4096, ErrRadiusTooLarge},
		{2, 5, 5, nil},
		{0, 1, 1, nil},
		// Border-clamped kernels may reach the image extent legally.
		{5, 5, 10, nil},
	}
	for _, tc := range cases {
		err := validateRadius(tc.radius, tc.w, tc.h)
		if tc.want == nil {
			if err != nil {
				t.Fatalf("radius %d on %dx%d unexpectedly rejected: %v", tc.radius, tc.w, tc.h, err)
			}
			continue
		}
		if !errors.Is(err, tc.want) {
			t.Fatalf("radius %d on %dx%d: got %v want %v", tc.radius, tc.w, tc.h, err, tc.want)
		}
	}
}

func TestNormalizeAngle(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{0, 0}, {90, 90}, {360, 0}, {450, 90},
		{-90, 270}, {-360, 0}, {-450, 270}, {30.5, 30.5},
	}
	for _, tc := range cases {
		got, err := normalizeAngle(tc.in)
		if err != nil {
			t.Fatalf("normalizeAngle(%v): %v", tc.in, err)
		}
		if math.Abs(got-tc.want) > 1e-9 {
			t.Fatalf("normalizeAngle(%v) = %v want %v", tc.in, got, tc.want)
		}
	}
	if _, err := normalizeAngle(math.NaN()); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("NaN angle: got %v", err)
	}
	if _, err := normalizeAngle(math.Inf(1)); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("Inf angle: got %v", err)
	}
}

func TestNearestSourceIndexRule(t *testing.T) {
	// src index must equal floor((dst+0.5)*src/dst), clamped.
	for _, tc := range []struct{ dstSize, srcSize int }{
		{4, 2}, {2, 4}, {3, 3}, {5, 7}, {7, 5}, {1, 4}, {4, 1}, {1, 1},
		{11, 13}, {13, 11},
	} {
		for d := 0; d < tc.dstSize; d++ {
			coord := (float64(d) + 0.5) * float64(tc.srcSize) / float64(tc.dstSize)
			want := int(math.Floor(coord))
			if want < 0 {
				want = 0
			}
			if want >= tc.srcSize {
				want = tc.srcSize - 1
			}
			if got := nearestSourceIndex(d, tc.dstSize, tc.srcSize); got != want {
				t.Fatalf("nearest(%d, %d->%d) = %d want %d", d, tc.dstSize, tc.srcSize, got, want)
			}
		}
	}
}

func TestBufferPoolReuse(t *testing.T) {
	var p BufferPool
	b1 := p.Get(128)
	if len(b1) != 128 {
		t.Fatalf("Get(128) len = %d", len(b1))
	}
	for i := range b1 {
		b1[i] = uint8(i)
	}
	p.Put(b1)
	b2 := p.Get(64)
	if cap(b2) < 128 {
		t.Fatalf("pool did not reuse buffer: cap = %d", cap(b2))
	}
	// Oversized buffers must be dropped, never retained.
	big := make([]uint8, poolMaxBytes+1)
	p.Put(big)
}

func TestAliasedDetection(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	// A view into the same backing array is an alias.
	view := &image.NRGBA{Pix: src.Pix, Stride: src.Stride, Rect: src.Rect}
	if !Aliased(src, view) {
		t.Fatal("shared Pix backing array not detected as alias")
	}
	independent := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	if Aliased(src, independent) {
		t.Fatal("independent buffers wrongly reported as aliased")
	}
	if Aliased(src, nil) {
		t.Fatal("nil dst wrongly reported as aliased")
	}
}

func TestTransformIntoRollback(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	fillNRGBA(src, color.NRGBA{10, 20, 30, 255})
	dst := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	before := snapshot(dst)

	// Aliased dst must be rejected and left untouched.
	view := &image.NRGBA{Pix: src.Pix, Stride: src.Stride, Rect: src.Rect}
	if err := TransformInto(view, src, func(img image.Image) (*image.NRGBA, error) {
		return nrgba(4, 4), nil
	}); !errors.Is(err, ErrAliasedImage) {
		t.Fatalf("alias: got %v want ErrAliasedImage", err)
	}
	if !compareBytes(view.Pix, src.Pix, 0) {
		t.Fatal("source modified after aliased write rejection")
	}

	// Failing transform must not touch dst.
	boom := errors.New("boom")
	err := TransformInto(dst, src, func(img image.Image) (*image.NRGBA, error) {
		return nil, boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("got %v want boom", err)
	}
	if !compareBytes(dst.Pix, before, 0) {
		t.Fatal("dst modified after failed transform")
	}

	// Size mismatch must be reported without writing.
	big := nrgba(5, 5)
	err = TransformInto(dst, src, func(img image.Image) (*image.NRGBA, error) { return big, nil })
	if !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("size mismatch: got %v", err)
	}
	if !compareBytes(dst.Pix, before, 0) {
		t.Fatal("dst modified after size mismatch")
	}

	// Successful path writes exactly the transform output.
	out := nrgba(4, 4)
	fillNRGBA(out, color.NRGBA{9, 8, 7, 6})
	if err := TransformInto(dst, src, func(img image.Image) (*image.NRGBA, error) { return out, nil }); err != nil {
		t.Fatalf("successful TransformInto: %v", err)
	}
	if !compareBytes(dst.Pix, out.Pix, 0) {
		t.Fatal("dst does not match transform output")
	}
	if !compareBytes(src.Pix, snapshot(func() *image.NRGBA {
		s := image.NewNRGBA(image.Rect(0, 0, 4, 4))
		fillNRGBA(s, color.NRGBA{10, 20, 30, 255})
		return s
	}()), 0) {
		t.Fatal("source modified by successful TransformInto")
	}
}

func TestResizeCheckedErrors(t *testing.T) {
	src := nrgba(4, 4)
	if _, err := ResizeChecked(nil, 2, 2, Linear); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil: %v", err)
	}
	if _, err := ResizeChecked(src, -1, 2, Linear); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("negative width: %v", err)
	}
	if _, err := ResizeChecked(src, 0, 0, Linear); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("zero both: %v", err)
	}
}

func TestResizeCheckedOddDimensionsAndNearestRule(t *testing.T) {
	// 5x3 -> 3x5 nearest neighbor; compare against an explicit reference
	// computed with the documented floor((d+0.5)*src/dst) rule.
	src := image.NewNRGBA(image.Rect(0, 0, 5, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 5; x++ {
			off := y*src.Stride + x*4
			src.Pix[off] = uint8(x * 50)
			src.Pix[off+1] = uint8(y * 80)
			src.Pix[off+2] = uint8(x*30 + y*10)
			src.Pix[off+3] = 255
		}
	}
	got, err := ResizeChecked(src, 3, 5, NearestNeighbor)
	if err != nil {
		t.Fatal(err)
	}
	want := image.NewNRGBA(image.Rect(0, 0, 3, 5))
	for y := 0; y < 5; y++ {
		sy := nearestSourceIndex(y, 5, 3)
		for x := 0; x < 3; x++ {
			sx := nearestSourceIndex(x, 3, 5)
			so := sy*src.Stride + sx*4
			do := y*want.Stride + x*4
			copy(want.Pix[do:do+4], src.Pix[so:so+4])
		}
	}
	if !compareNRGBA(got, want, 0) {
		t.Fatalf("nearest output differs from reference\n got %v\nwant %v", got.Pix, want.Pix)
	}
}

func TestResizeTwoPassEqualsOnePassReference(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 7, 5))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*37 + 11) & 0xff)
	}
	// Two successive resizes must match a reference built with the same
	// separable passes, pixel for pixel (no tolerance on channel order).
	mid, err := ResizeChecked(src, 11, 9, Linear)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ResizeChecked(mid, 6, 4, CatmullRom)
	if err != nil {
		t.Fatal(err)
	}
	ref1, err := resizeFilteredChecked(src, 11, 9, Linear)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := resizeFilteredChecked(ref1, 6, 4, CatmullRom)
	if err != nil {
		t.Fatal(err)
	}
	if !compareNRGBA(got, ref, 0) {
		t.Fatal("chained resizes do not match explicit reference path")
	}
}

func TestResizePreservesChannelOrder(t *testing.T) {
	// Pure red/green/blue pixels must never swap channels through resampling.
	src := image.NewNRGBA(image.Rect(0, 0, 3, 1))
	src.Pix = []uint8{
		255, 0, 0, 255,
		0, 255, 0, 255,
		0, 0, 255, 255,
	}
	got, err := ResizeChecked(src, 6, 1, Box)
	if err != nil {
		t.Fatal(err)
	}
	// Up-scaling pure primaries must keep every pixel a pure primary (only
	// one channel non-zero); any two-channel mixture would indicate a channel
	// ordering bug rather than normal interpolation.
	for x := 0; x < 6; x++ {
		off := x * 4
		r, g, b := got.Pix[off], got.Pix[off+1], got.Pix[off+2]
		nonZero := 0
		for _, v := range []uint8{r, g, b} {
			if v > 0 {
				nonZero++
			}
		}
		if nonZero > 1 {
			t.Fatalf("channel bleed at x=%d: %d,%d,%d", x, r, g, b)
		}
		if got.Pix[off+3] != 255 {
			t.Fatalf("alpha changed at x=%d: %d", x, got.Pix[off+3])
		}
	}
	// Positional order: red stays left, green middle, blue right.
	if got.Pix[0] == 0 {
		t.Fatalf("leftmost pixel is not red: %v", got.Pix[:4])
	}
	if got.Pix[5*4+2] == 0 {
		t.Fatalf("rightmost pixel is not blue: %v", got.Pix[5*4:5*4+4])
	}
}

func TestResizeAspectRatioPreservation(t *testing.T) {
	src := nrgba(6, 4)
	got, err := ResizeChecked(src, 3, 0, Box)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rect.Dx() != 3 || got.Rect.Dy() != 2 {
		t.Fatalf("aspect ratio: got %dx%d", got.Rect.Dx(), got.Rect.Dy())
	}
	got, err = ResizeChecked(src, 0, 2, Box)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rect.Dx() != 3 || got.Rect.Dy() != 2 {
		t.Fatalf("aspect ratio: got %dx%d", got.Rect.Dx(), got.Rect.Dy())
	}
}

func TestCropChecked(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 6, 4))
	for i := range src.Pix {
		src.Pix[i] = uint8(i)
	}
	got, err := CropChecked(src, image.Rect(1, 1, 4, 3))
	if err != nil {
		t.Fatal(err)
	}
	if got.Rect.Dx() != 3 || got.Rect.Dy() != 2 {
		t.Fatalf("crop size: %v", got.Rect)
	}
	// Explicit pixel check against source coordinates.
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			so := (y+1)*src.Stride + (x+1)*4
			do := y*got.Stride + x*4
			for k := 0; k < 4; k++ {
				if got.Pix[do+k] != src.Pix[so+k] {
					t.Fatalf("pixel mismatch at (%d,%d,%d)", x, y, k)
				}
			}
		}
	}
	if _, err := CropChecked(src, image.Rect(10, 10, 12, 12)); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("outside crop: got %v", err)
	}
	if _, err := CropChecked(src, image.Rectangle{}); !errors.Is(err, ErrInvalidSize) {
		t.Fatalf("empty rect: got %v", err)
	}
	if _, err := CropChecked(nil, image.Rect(0, 0, 1, 1)); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil crop: got %v", err)
	}
}

func TestRotateCheckedNegativeAndTransparent(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	for i := range src.Pix {
		src.Pix[i] = uint8(200)
	}
	// -90 must equal 270 and be an exact transpose-like mapping.
	neg, err := RotateChecked(src, -90, color.NRGBA{0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	pos, err := RotateChecked(src, 270, color.NRGBA{0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !compareNRGBA(neg, pos, 0) {
		t.Fatal("Rotate(-90) differs from Rotate(270)")
	}
	if neg.Rect.Dx() != 2 || neg.Rect.Dy() != 3 {
		t.Fatalf("rotated size: got %dx%d want 2x3", neg.Rect.Dx(), neg.Rect.Dy())
	}

	// Arbitrary angle with transparent background: corners must keep alpha 0
	// and carry zeroed RGB (no black fringe).
	angled, err := RotateChecked(src, 37, color.NRGBA{0, 0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	var transparentCorner color.NRGBA
	found := false
	for y := 0; y < angled.Rect.Dy() && !found; y++ {
		for x := 0; x < angled.Rect.Dx(); x++ {
			off := y*angled.Stride + x*4
			if angled.Pix[off+3] == 0 {
				transparentCorner = color.NRGBA{
					angled.Pix[off], angled.Pix[off+1], angled.Pix[off+2], angled.Pix[off+3],
				}
				found = true
				break
			}
		}
	}
	if !found {
		t.Fatal("expected at least one fully transparent corner pixel")
	}
	if transparentCorner != (color.NRGBA{0, 0, 0, 0}) {
		t.Fatalf("transparent pixel carries color data: %v", transparentCorner)
	}

	if _, err := RotateChecked(nil, 30, color.Black); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil rotate: %v", err)
	}
	if _, err := RotateChecked(src, math.NaN(), color.Black); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("NaN rotate: %v", err)
	}
	if _, err := RotateChecked(src, 30, nil); !errors.Is(err, ErrInvalidParameter) {
		t.Fatalf("nil bg color: %v", err)
	}
}

func TestRotateCardinalAnglesExact(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*7 + 3) & 0xff)
	}
	for _, tc := range []struct {
		angle float64
		ref   func(image.Image) *image.NRGBA
	}{
		{0, Clone},
		{90, Rotate90},
		{180, Rotate180},
		{270, Rotate270},
		{-270, Rotate90},
		{-180, Rotate180},
		{-90, Rotate270},
	} {
		got, err := RotateChecked(src, tc.angle, color.Black)
		if err != nil {
			t.Fatalf("angle %v: %v", tc.angle, err)
		}
		want := tc.ref(src)
		if !compareNRGBA(got, want, 0) {
			t.Fatalf("angle %v differs from legacy cardinal transform", tc.angle)
		}
	}
}

func TestSourceImageNeverMutated(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 6, 4))
	for i := range src.Pix {
		src.Pix[i] = uint8((i*13 + 5) & 0xff)
	}
	before := snapshot(src)
	ops := []func() (*image.NRGBA, error){
		func() (*image.NRGBA, error) { return ResizeChecked(src, 3, 3, Linear) },
		func() (*image.NRGBA, error) { return ResizeChecked(src, 12, 8, NearestNeighbor) },
		func() (*image.NRGBA, error) { return CropChecked(src, image.Rect(0, 0, 3, 2)) },
		func() (*image.NRGBA, error) { return RotateChecked(src, 33, color.NRGBA{1, 2, 3, 4}) },
		func() (*image.NRGBA, error) {
			return ApplyColorPipeline(src, ColorAdjustments{Brightness: 10, Contrast: -5, Gamma: 1.2, Saturation: 20})
		},
	}
	for i, op := range ops {
		if _, err := op(); err != nil {
			t.Fatalf("op %d: %v", i, err)
		}
		if !compareBytes(src.Pix, before, 0) {
			t.Fatalf("op %d mutated the source image", i)
		}
	}
}
