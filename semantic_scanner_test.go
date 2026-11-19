package imaging

import (
	"image"
	"testing"
)

// TestScannerRGBAUnpremultiply locks the exact 8-bit RGBA-to-NRGBA conversion.
// The scanner must undo premultiplication with integer truncation, matching
// the standard library, so that low coverage values never inflate to full
// intensity and edge pixels stay free of fringes.
func TestScannerRGBAUnpremultiply(t *testing.T) {
	cases := []struct {
		prem         uint8
		alpha        uint8
		wantStraight uint8
	}{
		{0, 0, 0},
		{1, 16, 15},   // 1*255/16 truncates to 15, never 239/255
		{16, 128, 31}, // 16*255/128 = 31.875 -> 31
		{128, 200, 163},
		{255, 255, 255},
	}
	img := image.NewRGBA(image.Rect(0, 0, len(cases), 1))
	for i, c := range cases {
		off := i * 4
		img.Pix[off], img.Pix[off+1], img.Pix[off+2], img.Pix[off+3] =
			c.prem, c.prem, c.prem, c.alpha
	}
	s := newScanner(img)
	buf := make([]uint8, len(cases)*4)
	s.scan(0, 0, len(cases), 1, buf)
	for i, c := range cases {
		off := i * 4
		for k := 0; k < 3; k++ {
			if buf[off+k] != c.wantStraight {
				t.Fatalf("case %d channel %d: got %d want %d (prem=%d a=%d)",
					i, k, buf[off+k], c.wantStraight, c.prem, c.alpha)
			}
		}
		if buf[off+3] != c.alpha {
			t.Fatalf("case %d alpha: got %d want %d", i, buf[off+3], c.alpha)
		}
	}
}

// TestScannerTransparentRGBAIsZero verifies that fully transparent premultiplied
// pixels scan to (0,0,0,0).
func TestScannerTransparentRGBAIsZero(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Pix = []uint8{99, 99, 99, 0} // color data must be discarded
	s := newScanner(img)
	buf := make([]uint8, 4)
	s.scan(0, 0, 1, 1, buf)
	if buf[0] != 0 || buf[1] != 0 || buf[2] != 0 || buf[3] != 0 {
		t.Fatalf("transparent RGBA scanned to %v, want all zero", buf)
	}
}
