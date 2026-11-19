package imaging

import (
	"image"
	"image/color"
)

// This file defines the alpha handling of the semantic processing path.
//
// All images produced by the package are non-premultiplied *image.NRGBA.
// Whenever an operation needs premultiplied arithmetic (resampling,
// rotation, compositing) it must convert through the helpers here so that:
//
//   - fully transparent pixels carry (0, 0, 0, 0), never stale color data that
//     would bleed into visible edges (dark halos);
//   - rounding is identical in every direction: scale by an alpha coverage is
//     round-half-up in the integer domain, matching clamp for float paths;
//   - the conversions are characterized by tests (exact at alpha 0 and 255,
//     at most one quantization step elsewhere) instead of relying on implicit
//     image/color model behavior.

// mul8 scales the channel value v by the alpha coverage a/255 using
// round-half-up integer arithmetic (the numerator is an integer so the result
// is equivalent to floor(x + 0.5)).
func mul8(v, a uint8) uint8 {
	return uint8((uint16(v)*uint16(a) + 127) / 255)
}

// div8 reverses mul8: it scales v back from premultiplied coverage to full
// intensity given alpha a. The caller must ensure a > 0. Integer truncation
// (floor for the non-negative values here) matches the standard library
// RGBA-to-NRGBA conversion and prevents a tiny coverage value from inflating
// to full intensity; mul8's round-half-up compensates on the forward path so
// visible edge pixels never go dark.
func div8(v, a uint8) uint8 {
	return uint8(uint16(v) * 255 / uint16(a))
}

// Premultiply converts a non-premultiplied NRGBA image into a premultiplied
// RGBA image. Fully transparent pixels are stored as (0, 0, 0, 0) regardless
// of their input RGB values, which keeps transparent regions free of color
// contamination.
func Premultiply(img image.Image) (*image.RGBA, error) {
	if _, err := validateImage(img); err != nil {
		return nil, err
	}
	src := newScanner(img)
	dst := image.NewRGBA(image.Rect(0, 0, src.w, src.h))
	parallel(0, src.h, func(ys <-chan int) {
		scanLine := sharedBufferPool.Get(src.w * 4)
		defer sharedBufferPool.Put(scanLine)
		for y := range ys {
			src.scan(0, y, src.w, y+1, scanLine)
			off := y * dst.Stride
			for x := 0; x < src.w*4; x += 4 {
				a := scanLine[x+3]
				if a == 0 {
					dst.Pix[off], dst.Pix[off+1], dst.Pix[off+2], dst.Pix[off+3] = 0, 0, 0, 0
				} else if a == 0xff {
					copy(dst.Pix[off:off+4], scanLine[x:x+4])
				} else {
					dst.Pix[off] = mul8(scanLine[x], a)
					dst.Pix[off+1] = mul8(scanLine[x+1], a)
					dst.Pix[off+2] = mul8(scanLine[x+2], a)
					dst.Pix[off+3] = a
				}
				off += 4
			}
		}
	})
	return dst, nil
}

// Unpremultiply converts a premultiplied RGBA image back into non-premultiplied
// NRGBA. Pixels with alpha 0 become (0, 0, 0, 0) and opaque pixels keep their
// channels exactly; intermediate alpha values use integer truncation. The
// scanner already performs exactly one premultiplied-to-straight conversion,
// so this function must not apply a second division on the result.
func Unpremultiply(img image.Image) (*image.NRGBA, error) {
	if _, err := validateImage(img); err != nil {
		return nil, err
	}
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, src.w, src.h))
	parallel(0, src.h, func(ys <-chan int) {
		scanLine := sharedBufferPool.Get(src.w * 4)
		defer sharedBufferPool.Put(scanLine)
		for y := range ys {
			src.scan(0, y, src.w, y+1, scanLine)
			off := y * dst.Stride
			copy(dst.Pix[off:off+src.w*4], scanLine)
		}
	})
	return dst, nil
}

// CompositeOver draws src over background using the Porter-Duff source-over
// operator evaluated in the premultiplied domain. Working on premultiplied
// values means partially transparent edge pixels blend without introducing
// dark halos: a fully transparent source pixel leaves the background
// untouched and never contributes black.
func CompositeOver(background, src image.Image) (*image.NRGBA, error) {
	bgBounds, err := validateImage(background)
	if err != nil {
		return nil, err
	}
	if _, err := validateImage(src); err != nil {
		return nil, err
	}
	if !src.Bounds().Size().Eq(bgBounds.Size()) {
		return nil, ErrSizeMismatch
	}
	bg := newScanner(background)
	fg := newScanner(src)
	w, h := bg.w, bg.h
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))

	parallel(0, h, func(ys <-chan int) {
		bgLine := sharedBufferPool.Get(w * 4)
		fgLine := sharedBufferPool.Get(w * 4)
		defer sharedBufferPool.Put(bgLine)
		defer sharedBufferPool.Put(fgLine)
		for y := range ys {
			bg.scan(0, y, w, y+1, bgLine)
			fg.scan(0, y, w, y+1, fgLine)
			off := y * dst.Stride
			for x := 0; x < w; x++ {
				i := x * 4
				sa := fgLine[i+3]
				if sa == 0 {
					copy(dst.Pix[off:off+4], bgLine[i:i+4])
					off += 4
					continue
				}
				if sa == 0xff {
					copy(dst.Pix[off:off+4], fgLine[i:i+4])
					off += 4
					continue
				}
				da := bgLine[i+3]
				// Premultiplied source-over: out = src + dst*(1-sa/255).
				outA := uint16(sa) + uint16(da)*(255-uint16(sa))/255
				if outA == 0 {
					dst.Pix[off], dst.Pix[off+1], dst.Pix[off+2], dst.Pix[off+3] = 0, 0, 0, 0
					off += 4
					continue
				}
				inv := 255 - uint16(sa)
				for c := 0; c < 3; c++ {
					srcPrem := uint16(mul8(fgLine[i+c], sa))
					dstPrem := uint16(mul8(bgLine[i+c], da)) * inv / 255
					v := (srcPrem + dstPrem) * 255 / outA
					if v > 255 {
						v = 255
					}
					dst.Pix[off+c] = uint8(v)
				}
				dst.Pix[off+3] = uint8(outA)
				off += 4
			}
		}
	})
	return dst, nil
}

// interpolatePremultiplied is the bilinear sampling primitive shared by
// geometric transforms over transparent content. Inputs are non-premultiplied
// NRGBA; weights are applied to premultiplied channels and the result is
// unpremultiplied only at storage time, so a weight window spanning
// transparent pixels never darkens the visible edge. bg is the fill color for
// out-of-bounds sample positions.
func interpolatePremultiplied(src *image.NRGBA, x0, y0 int, xq, yq float64, bg color.NRGBA) (r, g, b, a uint8) {
	weights := [4]float64{
		(1 - xq) * (1 - yq),
		xq * (1 - yq),
		(1 - xq) * yq,
		xq * yq,
	}
	points := [4][2]int{{x0, y0}, {x0 + 1, y0}, {x0, y0 + 1}, {x0 + 1, y0 + 1}}
	bounds := src.Bounds()

	var pr, pg, pb, pa float64
	for i := 0; i < 4; i++ {
		w := weights[i]
		var cr, cg, cb, ca uint8
		p := image.Pt(points[i][0], points[i][1])
		if p.In(bounds) {
			off := p.Y*src.Stride + p.X*4
			cr, cg, cb, ca = src.Pix[off], src.Pix[off+1], src.Pix[off+2], src.Pix[off+3]
		} else {
			cr, cg, cb, ca = bg.R, bg.G, bg.B, bg.A
		}
		pr += float64(mul8(cr, ca)) * w
		pg += float64(mul8(cg, ca)) * w
		pb += float64(mul8(cb, ca)) * w
		pa += float64(ca) * w
	}
	if pa <= 0 {
		return 0, 0, 0, 0
	}
	a = clamp(pa)
	r = clamp(pr * 255.0 / pa)
	g = clamp(pg * 255.0 / pa)
	b = clamp(pb * 255.0 / pa)
	return r, g, b, a
}
