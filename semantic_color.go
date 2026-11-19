package imaging

import (
	"image"
	"math"
)

// This file implements the reproducible color adjustment pipeline of the
// semantic processing path.
//
// The pipeline order is fixed and documented:
//
//  1. Brightness  (additive shift in 0..255 units)
//  2. Contrast    (scale around the midpoint 127.5)
//  3. Gamma       (power-law transfer)
//  4. Saturation  (luminance interpolation in HSL space)
//
// The first three stages operate on float64 channel values in the 0..255
// range; saturation is the only stage that converts to normalized 0..1 RGB
// values and back. Every conversion between float and uint8 goes through the
// single truncation point roundToUint8 (round-half-up, then clamp), which is
// the integer counterpart of clamp. Alpha is never altered, and all pixel
// storage is freshly allocated: the source image is never written to.

// ColorAdjustments configures the four-stage color pipeline. Zero-valued
// fields are identity operations.
type ColorAdjustments struct {
	// Brightness shifts every channel by the given percentage of 255,
	// clamped to [-100, 100].
	Brightness float64

	// Contrast scales around the midpoint. Range [-100, 100]; -100 produces
	// uniform mid-gray.
	Contrast float64

	// Gamma must be positive; 1.0 is identity.
	Gamma float64

	// Saturation ranges from -100 (grayscale) to 100 (doubled saturation).
	Saturation float64
}

// Valid reports whether every field is inside its documented range.
func (a ColorAdjustments) Valid() bool {
	return inPercentRange(a.Brightness) &&
		inPercentRange(a.Contrast) &&
		inPercentRange(a.Saturation) &&
		(a.Gamma == 0 || (a.Gamma > 0 && !isNonFinite(a.Gamma)))
}

func inPercentRange(v float64) bool {
	return v >= -100 && v <= 100 && !isNonFinite(v)
}

func isNonFinite(v float64) bool {
	return math.IsNaN(v) || math.IsInf(v, 0)
}

// IsIdentity reports whether applying the adjustments leaves every pixel
// unchanged.
func (a ColorAdjustments) IsIdentity() bool {
	return a.Brightness == 0 && a.Contrast == 0 && a.Saturation == 0 &&
		(a.Gamma == 1.0 || a.Gamma == 0)
}

// roundToUint8 is the single integer truncation point of the color pipeline:
// round half away from zero (half up for the non-negative channel values) and
// clamp to the uint8 range.
func roundToUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// ApplyColorPipeline runs the four documented stages in the fixed order and
// returns a freshly allocated image. It returns ErrInvalidParameter if the
// configuration is invalid and ErrEmptyImage/ErrNilImage for a bad source.
func ApplyColorPipeline(img image.Image, adj ColorAdjustments) (*image.NRGBA, error) {
	if _, err := validateImage(img); err != nil {
		return nil, err
	}
	if !adj.Valid() {
		return nil, ErrInvalidParameter
	}

	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, src.w, src.h))

	// Precompute stage parameters once; the order of the rows below defines
	// the reproducible pipeline order.
	brightnessShift := 255.0 * adj.Brightness / 100.0

	contrastFactor := 1.0
	if adj.Contrast != 0 {
		v := (100.0 + adj.Contrast) / 100.0
		if v <= 1 {
			contrastFactor = v
		} else {
			contrastFactor = 1 / (2 - v)
		}
	}

	gammaExponent := 0.0
	if adj.Gamma > 0 {
		gammaExponent = 1.0 / adj.Gamma
	}

	saturationMultiplier := 1.0
	if adj.Saturation != 0 {
		saturationMultiplier = 1 + adj.Saturation/100
	}

	parallel(0, src.h, func(ys <-chan int) {
		scanLine := sharedBufferPool.Get(src.w * 4)
		defer sharedBufferPool.Put(scanLine)
		for y := range ys {
			src.scan(0, y, src.w, y+1, scanLine)
			off := y * dst.Stride
			for x := 0; x < src.w; x++ {
				i := x * 4
				r := float64(scanLine[i])
				g := float64(scanLine[i+1])
				b := float64(scanLine[i+2])
				a := scanLine[i+3]

				// Stage 1: brightness.
				if adj.Brightness != 0 {
					r += brightnessShift
					g += brightnessShift
					b += brightnessShift
				}

				// Stage 2: contrast around 127.5.
				if adj.Contrast != 0 {
					r = 127.5 + (r-127.5)*contrastFactor
					g = 127.5 + (g-127.5)*contrastFactor
					b = 127.5 + (b-127.5)*contrastFactor
				}

				// Stage 3: gamma on normalized values.
				if adj.Gamma > 0 && adj.Gamma != 1 {
					r = clampFloat(r) / 255.0
					g = clampFloat(g) / 255.0
					b = clampFloat(b) / 255.0
					r = pow255(r, gammaExponent)
					g = pow255(g, gammaExponent)
					b = pow255(b, gammaExponent)
				}

				// Stage 4: saturation via HSL luminance interpolation.
				if adj.Saturation != 0 {
					var h, s, l float64
					h, s, l = rgbToHSL(roundToUint8(r), roundToUint8(g), roundToUint8(b))
					s *= saturationMultiplier
					if s > 1 {
						s = 1
					}
					nr, ng, nb := hslToRGB(h, s, l)
					dst.Pix[off], dst.Pix[off+1], dst.Pix[off+2] = nr, ng, nb
				} else {
					dst.Pix[off] = roundToUint8(r)
					dst.Pix[off+1] = roundToUint8(g)
					dst.Pix[off+2] = roundToUint8(b)
				}
				dst.Pix[off+3] = a
				off += 4
			}
		}
	})
	return dst, nil
}

func clampFloat(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

func pow255(normalized, exponent float64) float64 {
	return float64(clamp(math.Pow(normalized, exponent) * 255.0))
}

// BrightnessChecked is the error-returning brightness adjustment.
func BrightnessChecked(img image.Image, percentage float64) (*image.NRGBA, error) {
	if !inPercentRange(percentage) {
		return nil, ErrInvalidParameter
	}
	return ApplyColorPipeline(img, ColorAdjustments{Brightness: percentage})
}

// ContrastChecked is the error-returning contrast adjustment.
func ContrastChecked(img image.Image, percentage float64) (*image.NRGBA, error) {
	if !inPercentRange(percentage) {
		return nil, ErrInvalidParameter
	}
	return ApplyColorPipeline(img, ColorAdjustments{Contrast: percentage})
}

// GammaChecked is the error-returning gamma adjustment.
func GammaChecked(img image.Image, gamma float64) (*image.NRGBA, error) {
	if gamma <= 0 || isNonFinite(gamma) {
		return nil, ErrInvalidParameter
	}
	return ApplyColorPipeline(img, ColorAdjustments{Gamma: gamma})
}

// SaturationChecked is the error-returning saturation adjustment.
func SaturationChecked(img image.Image, percentage float64) (*image.NRGBA, error) {
	if !inPercentRange(percentage) {
		return nil, ErrInvalidParameter
	}
	return ApplyColorPipeline(img, ColorAdjustments{Saturation: percentage})
}
