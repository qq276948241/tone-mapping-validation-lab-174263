package imaging

import (
	"image"
	"image/color"
	"math"
)

// This file contains the geometric transforms of the semantic processing path.
//
// Coordinate conventions (fixed, do not change):
//
//   - Pixel (x, y) occupies the unit square [x, x+1) x [y, y+1); its center
//     is at (x+0.5, y+0.5).
//   - Nearest-neighbor sampling maps each destination pixel center back to the
//     source with src = (dst+0.5)*srcSize/dstSize and selects floor(src),
//     clamped to the source extent. The single implementation of this rule is
//     nearestSourceIndex.
//   - Filtered (area/cubic/lanczos) sampling uses half-pixel centered
//     coordinates (precomputeWeights) with round-half-up clamping (clamp) when
//     storing uint8 values.
//   - Rotation is counter-clockwise, angles are normalized to [0, 360), and
//     uncovered pixels are filled with the caller-provided background color,
//     including a fully transparent background.
//
// Every function returns a freshly allocated *image.NRGBA image; the source is
// never modified.

// ResizeChecked is the error-returning counterpart of Resize. If one of width
// or height is 0 the aspect ratio is preserved; both zero is an error. The
// filter selects the fixed sampling rule: NearestNeighbor (Support <= 0) uses
// nearestSourceIndex, every other filter uses the half-pixel centered
// separable resampler.
func ResizeChecked(img image.Image, width, height int, filter ResampleFilter) (*image.NRGBA, error) {
	bounds, err := validateImage(img)
	if err != nil {
		return nil, err
	}
	if width < 0 || height < 0 || (width == 0 && height == 0) {
		return nil, ErrInvalidSize
	}

	srcW, srcH := bounds.Dx(), bounds.Dy()
	dstW, dstH := width, height
	if dstW == 0 {
		dstW = int(math.Max(1, math.Floor(float64(dstH)*float64(srcW)/float64(srcH)+0.5)))
	}
	if dstH == 0 {
		dstH = int(math.Max(1, math.Floor(float64(dstW)*float64(srcH)/float64(srcW)+0.5)))
	}
	if err := validateSize(dstW, dstH); err != nil {
		return nil, err
	}

	if srcW == dstW && srcH == dstH {
		return Clone(img), nil
	}

	if filter.Support <= 0 {
		return resizeNearestChecked(img, dstW, dstH)
	}
	return resizeFilteredChecked(img, dstW, dstH, filter)
}

// resizeNearestChecked implements the fixed nearest-neighbor rule. Each
// goroutine owns a scratch pixel buffer acquired from sharedBufferPool, so the
// function is safe for concurrent invocation.
func resizeNearestChecked(img image.Image, width, height int) (*image.NRGBA, error) {
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, width, height))
	dx := src.w
	dy := src.h

	parallel(0, height, func(ys <-chan int) {
		buf := sharedBufferPool.Get(4)
		defer sharedBufferPool.Put(buf)
		for y := range ys {
			srcY := nearestSourceIndex(y, height, dy)
			dstOff := y * dst.Stride
			for x := 0; x < width; x++ {
				srcX := nearestSourceIndex(x, width, dx)
				src.scan(srcX, srcY, srcX+1, srcY+1, buf)
				copy(dst.Pix[dstOff:dstOff+4], buf)
				dstOff += 4
			}
		}
	})
	return dst, nil
}

// resizeFilteredChecked runs the documented separable resampling path:
// horizontal pass first, then vertical, sharing the exact same precomputed
// weight tables as the reference implementation used by the test suite.
func resizeFilteredChecked(img image.Image, width, height int, filter ResampleFilter) (*image.NRGBA, error) {
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	if srcW != width && srcH != height {
		first, err := resizeHorizontalChecked(img, width, filter)
		if err != nil {
			return nil, err
		}
		return resizeVerticalChecked(first, height, filter)
	}
	if srcW != width {
		return resizeHorizontalChecked(img, width, filter)
	}
	return resizeVerticalChecked(img, height, filter)
}

func resizeHorizontalChecked(img image.Image, width int, filter ResampleFilter) (*image.NRGBA, error) {
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, width, src.h))
	weights := precomputeWeights(width, src.w, filter)
	parallel(0, src.h, func(ys <-chan int) {
		scanLine := sharedBufferPool.Get(src.w * 4)
		defer sharedBufferPool.Put(scanLine)
		for y := range ys {
			src.scan(0, y, src.w, y+1, scanLine)
			j0 := y * dst.Stride
			for x := range weights {
				var r, g, b, a float64
				for _, w := range weights[x] {
					i := w.index * 4
					s := scanLine[i : i+4 : i+4]
					aw := float64(s[3]) * w.weight
					r += float64(s[0]) * aw
					g += float64(s[1]) * aw
					b += float64(s[2]) * aw
					a += aw
				}
				if a != 0 {
					aInv := 1 / a
					j := j0 + x*4
					d := dst.Pix[j : j+4 : j+4]
					d[0] = clamp(r * aInv)
					d[1] = clamp(g * aInv)
					d[2] = clamp(b * aInv)
					d[3] = clamp(a)
				}
			}
		}
	})
	return dst, nil
}

func resizeVerticalChecked(img image.Image, height int, filter ResampleFilter) (*image.NRGBA, error) {
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, src.w, height))
	weights := precomputeWeights(height, src.h, filter)
	parallel(0, src.w, func(xs <-chan int) {
		scanLine := sharedBufferPool.Get(src.h * 4)
		defer sharedBufferPool.Put(scanLine)
		for x := range xs {
			src.scan(x, 0, x+1, src.h, scanLine)
			for y := range weights {
				var r, g, b, a float64
				for _, w := range weights[y] {
					i := w.index * 4
					s := scanLine[i : i+4 : i+4]
					aw := float64(s[3]) * w.weight
					r += float64(s[0]) * aw
					g += float64(s[1]) * aw
					b += float64(s[2]) * aw
					a += aw
				}
				if a != 0 {
					aInv := 1 / a
					j := y*dst.Stride + x*4
					d := dst.Pix[j : j+4 : j+4]
					d[0] = clamp(r * aInv)
					d[1] = clamp(g * aInv)
					d[2] = clamp(b * aInv)
					d[3] = clamp(a)
				}
			}
		}
	})
	return dst, nil
}

// CropChecked is the error-returning counterpart of Crop. The rectangle is
// intersected with the image bounds; an empty intersection (or an empty
// rectangle) fails with ErrInvalidSize instead of silently returning an empty
// image.
func CropChecked(img image.Image, rect image.Rectangle) (*image.NRGBA, error) {
	bounds, err := validateImage(img)
	if err != nil {
		return nil, err
	}
	if rect.Empty() {
		return nil, ErrInvalidSize
	}
	r := rect.Intersect(bounds).Sub(bounds.Min)
	if r.Empty() {
		return nil, ErrInvalidSize
	}
	if r.Eq(bounds.Sub(bounds.Min)) {
		return Clone(img), nil
	}
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	rowSize := r.Dx() * 4
	parallel(r.Min.Y, r.Max.Y, func(ys <-chan int) {
		buf := sharedBufferPool.Get(rowSize)
		defer sharedBufferPool.Put(buf)
		for y := range ys {
			i := (y - r.Min.Y) * dst.Stride
			src.scan(r.Min.X, y, r.Max.X, y+1, buf)
			copy(dst.Pix[i:i+rowSize], buf)
		}
	})
	return dst, nil
}

// CropAnchorChecked crops a width x height rectangle positioned by anchor. It
// returns ErrInvalidSize when the requested size is not positive.
func CropAnchorChecked(img image.Image, width, height int, anchor Anchor) (*image.NRGBA, error) {
	bounds, err := validateImage(img)
	if err != nil {
		return nil, err
	}
	if width <= 0 || height <= 0 || width > MaxImageDimension || height > MaxImageDimension {
		return nil, ErrInvalidSize
	}
	pt := anchorPt(bounds, width, height, anchor)
	r := image.Rect(0, 0, width, height).Add(pt)
	b := bounds.Intersect(r)
	return CropChecked(img, b)
}

// RotateChecked is the error-returning counterpart of Rotate. Negative angles
// and angles outside [0, 360) are normalized; NaN/Inf return
// ErrInvalidParameter. The background color fills uncovered pixels and may be
// fully transparent (alpha 0), in which case the uncovered zone keeps alpha 0.
func RotateChecked(img image.Image, angle float64, bgColor color.Color) (*image.NRGBA, error) {
	angle, err := normalizeAngle(angle)
	if err != nil {
		return nil, err
	}
	if _, err := validateImage(img); err != nil {
		return nil, err
	}
	if bgColor == nil {
		return nil, ErrInvalidParameter
	}

	switch angle {
	case 0:
		return Clone(img), nil
	case 90:
		return Rotate90(img), nil
	case 180:
		return Rotate180(img), nil
	case 270:
		return Rotate270(img), nil
	}

	src := toNRGBA(img)
	srcW := src.Bounds().Max.X
	srcH := src.Bounds().Max.Y
	dstW, dstH := rotatedSize(srcW, srcH, angle)
	if err := validateSize(dstW, dstH); err != nil {
		return nil, err
	}
	dst := image.NewNRGBA(image.Rect(0, 0, dstW, dstH))

	srcXOff := float64(srcW)/2 - 0.5
	srcYOff := float64(srcH)/2 - 0.5
	dstXOff := float64(dstW)/2 - 0.5
	dstYOff := float64(dstH)/2 - 0.5

	bgColorNRGBA := color.NRGBAModel.Convert(bgColor).(color.NRGBA)
	sin, cos := math.Sincos(math.Pi * angle / 180)

	parallel(0, dstH, func(ys <-chan int) {
		for dstY := range ys {
			for dstX := 0; dstX < dstW; dstX++ {
				xf, yf := rotatePoint(float64(dstX)-dstXOff, float64(dstY)-dstYOff, sin, cos)
				xf, yf = xf+srcXOff, yf+srcYOff
				x0 := int(math.Floor(xf))
				y0 := int(math.Floor(yf))
				bounds := src.Bounds()
				j := dstY*dst.Stride + dstX*4
				// Fully outside the sampling support: paint the background
				// color directly, preserving a transparent background.
				if !image.Pt(x0, y0).In(image.Rect(bounds.Min.X-1, bounds.Min.Y-1, bounds.Max.X, bounds.Max.Y)) {
					dst.Pix[j], dst.Pix[j+1], dst.Pix[j+2], dst.Pix[j+3] = bgColorNRGBA.R, bgColorNRGBA.G, bgColorNRGBA.B, bgColorNRGBA.A
					continue
				}
				r, g, b, a := interpolatePremultiplied(src, x0, y0, xf-float64(x0), yf-float64(y0), bgColorNRGBA)
				dst.Pix[j], dst.Pix[j+1], dst.Pix[j+2], dst.Pix[j+3] = r, g, b, a
			}
		}
	})
	return dst, nil
}
