package imaging

import (
	"image"
	"math"
	"sync"
)

// This file contains the histogram, convolution and edge processing functions
// of the semantic processing path. They share the same input validation
// (validateImage), the same kernel-radius limits (validateRadius /
// MaxKernelRadius) and the same concurrency-safe scratch buffer management
// (sharedBufferPool). Every result is allocated independently of the source
// and all validation happens before any pixel is read for writing, so a
// failure can never modify the caller's original image.

// HistogramChecked is the error-returning counterpart of Histogram. The
// returned histogram is normalized (the 256 bins sum to 1.0). A nil image is
// rejected; note that an empty image is already rejected by validateImage,
// unlike the legacy function which silently returns zeros.
func HistogramChecked(img image.Image) ([256]float64, error) {
	if _, err := validateImage(img); err != nil {
		return [256]float64{}, err
	}

	var mu sync.Mutex
	var histogram [256]float64
	var total float64

	src := newScanner(img)
	parallel(0, src.h, func(ys <-chan int) {
		var tmpHistogram [256]float64
		var tmpTotal float64
		scanLine := sharedBufferPool.Get(src.w * 4)
		defer sharedBufferPool.Put(scanLine)
		for y := range ys {
			src.scan(0, y, src.w, y+1, scanLine)
			i := 0
			for x := 0; x < src.w; x++ {
				r := scanLine[i]
				g := scanLine[i+1]
				b := scanLine[i+2]
				lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
				tmpHistogram[roundToUint8(lum)]++
				tmpTotal++
				i += 4
			}
		}
		mu.Lock()
		for i := 0; i < 256; i++ {
			histogram[i] += tmpHistogram[i]
		}
		total += tmpTotal
		mu.Unlock()
	})

	if total <= 0 {
		return histogram, ErrEmptyImage
	}
	for i := 0; i < 256; i++ {
		histogram[i] /= total
	}
	return histogram, nil
}

// ConvolveChecked applies a square kernel of side 2*radius+1 to the image.
// radius 1 yields a 3x3 kernel, radius 2 a 5x5 kernel. The kernel length must
// match (2*radius+1)^2 or ErrInvalidParameter is returned. Edge pixels use
// clamp-to-border sampling. Options may be nil for default behavior.
func ConvolveChecked(img image.Image, kernel []float64, radius int, options *ConvolveOptions) (*image.NRGBA, error) {
	bounds, err := validateImage(img)
	if err != nil {
		return nil, err
	}
	if radius < 0 {
		return nil, ErrInvalidParameter
	}
	side := 2*radius + 1
	if len(kernel) != side*side {
		return nil, ErrInvalidParameter
	}
	for _, v := range kernel {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, ErrInvalidParameter
		}
	}
	if err := validateRadius(radius, bounds.Dx(), bounds.Dy()); err != nil {
		return nil, err
	}

	if options == nil {
		options = &ConvolveOptions{}
	}
	// Work on a copy of the kernel so normalization never mutates the
	// caller's slice.
	k := make([]float64, len(kernel))
	copy(k, kernel)
	if options.Normalize {
		normalizeKernel(k)
	}

	type coef struct {
		x, y int
		v    float64
	}
	coefs := make([]coef, 0, len(k))
	i := 0
	for y := -radius; y <= radius; y++ {
		for x := -radius; x <= radius; x++ {
			if k[i] != 0 {
				coefs = append(coefs, coef{x: x, y: y, v: k[i]})
			}
			i++
		}
	}

	src := toNRGBA(img)
	w, h := bounds.Dx(), bounds.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))

	parallel(0, h, func(ys <-chan int) {
		for y := range ys {
			for x := 0; x < w; x++ {
				var r, g, b float64
				for _, c := range coefs {
					ix := clampInt(x+c.x, 0, w-1)
					iy := clampInt(y+c.y, 0, h-1)
					off := iy*src.Stride + ix*4
					r += float64(src.Pix[off]) * c.v
					g += float64(src.Pix[off+1]) * c.v
					b += float64(src.Pix[off+2]) * c.v
				}
				if options.Abs {
					r = math.Abs(r)
					g = math.Abs(g)
					b = math.Abs(b)
				}
				if options.Bias != 0 {
					r += float64(options.Bias)
					g += float64(options.Bias)
					b += float64(options.Bias)
				}
				dstOff := y*dst.Stride + x*4
				dst.Pix[dstOff] = clamp(r)
				dst.Pix[dstOff+1] = clamp(g)
				dst.Pix[dstOff+2] = clamp(b)
				dst.Pix[dstOff+3] = src.Pix[y*src.Stride+x*4+3]
			}
		}
	})
	return dst, nil
}

// Convolve3x3Checked is the error-returning counterpart of Convolve3x3.
func Convolve3x3Checked(img image.Image, kernel [9]float64, options *ConvolveOptions) (*image.NRGBA, error) {
	return ConvolveChecked(img, kernel[:], 1, options)
}

// Convolve5x5Checked is the error-returning counterpart of Convolve5x5.
func Convolve5x5Checked(img image.Image, kernel [25]float64, options *ConvolveOptions) (*image.NRGBA, error) {
	return ConvolveChecked(img, kernel[:], 2, options)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// BlurChecked applies a separable Gaussian blur. Sigma must be positive and
// finite; the derived radius (ceil(sigma*3)) is checked against
// MaxKernelRadius, so extreme values such as math.MaxFloat64 fail with
// ErrRadiusTooLarge instead of allocating huge buffers.
func BlurChecked(img image.Image, sigma float64) (*image.NRGBA, error) {
	if _, err := validateImage(img); err != nil {
		return nil, err
	}
	if isNonFinite(sigma) {
		return nil, ErrInvalidParameter
	}
	if sigma <= 0 {
		return nil, ErrInvalidParameter
	}
	// Compare in floating point before any int conversion so an extreme
	// finite sigma cannot overflow the radius integer.
	if r := math.Ceil(sigma * 3.0); r > float64(MaxKernelRadius) {
		return nil, ErrRadiusTooLarge
	}
	radius := int(math.Ceil(sigma * 3.0))
	if err := validateRadius(radius, img.Bounds().Dx(), img.Bounds().Dy()); err != nil {
		return nil, err
	}

	kernel := make([]float64, radius+1)
	for i := 0; i <= radius; i++ {
		kernel[i] = gaussianBlurKernel(float64(i), sigma)
	}
	return blurSeparableChecked(img, kernel)
}

// BlurRadiusChecked is identical to BlurChecked but takes an explicit integer
// radius, making the sampling window deterministic for reference tests. The
// radius must be strictly smaller than both image dimensions.
func BlurRadiusChecked(img image.Image, radius int) (*image.NRGBA, error) {
	bounds, err := validateImage(img)
	if err != nil {
		return nil, err
	}
	if radius < 0 {
		return nil, ErrInvalidParameter
	}
	if radius >= bounds.Dx() || radius >= bounds.Dy() {
		return nil, ErrRadiusTooLarge
	}
	if err := validateRadius(radius, bounds.Dx(), bounds.Dy()); err != nil {
		return nil, err
	}
	sigma := float64(radius) / 3.0
	if sigma <= 0 {
		sigma = 1.0 / 3.0
	}
	kernel := make([]float64, radius+1)
	for i := 0; i <= radius; i++ {
		kernel[i] = gaussianBlurKernel(float64(i), sigma)
	}
	return blurSeparableChecked(img, kernel)
}

// blurSeparableChecked performs the horizontal then vertical pass. Both passes
// operate on premultiplied contributions and renormalize by alpha coverage, so
// transparent borders never darken neighboring pixels.
func blurSeparableChecked(img image.Image, kernel []float64) (*image.NRGBA, error) {
	radius := len(kernel) - 1
	src := newScanner(img)
	w, h := src.w, src.h
	tmp := image.NewNRGBA(image.Rect(0, 0, w, h))
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))

	// Horizontal pass.
	parallel(0, h, func(ys <-chan int) {
		scanLine := sharedBufferPool.Get(w * 4)
		defer sharedBufferPool.Put(scanLine)
		for y := range ys {
			src.scan(0, y, w, y+1, scanLine)
			for x := 0; x < w; x++ {
				lo := clampInt(x-radius, 0, w-1)
				hi := clampInt(x+radius, 0, w-1)
				var pr, pg, pb, pa float64
				for ix := lo; ix <= hi; ix++ {
					weight := kernel[absint(x-ix)]
					i := ix * 4
					a := scanLine[i+3]
					pr += float64(mul8(scanLine[i], a)) * weight
					pg += float64(mul8(scanLine[i+1], a)) * weight
					pb += float64(mul8(scanLine[i+2], a)) * weight
					pa += float64(a) * weight
				}
				off := y*tmp.Stride + x*4
				if pa <= 0 {
					tmp.Pix[off], tmp.Pix[off+1], tmp.Pix[off+2], tmp.Pix[off+3] = 0, 0, 0, 0
					continue
				}
				tmp.Pix[off] = clamp(pr * 255 / pa)
				tmp.Pix[off+1] = clamp(pg * 255 / pa)
				tmp.Pix[off+2] = clamp(pb * 255 / pa)
				tmp.Pix[off+3] = clamp(pa)
			}
		}
	})

	// Vertical pass.
	parallel(0, w, func(xs <-chan int) {
		column := sharedBufferPool.Get(h * 4)
		defer sharedBufferPool.Put(column)
		for x := range xs {
			for y := 0; y < h; y++ {
				off := y*tmp.Stride + x*4
				copy(column[y*4:y*4+4], tmp.Pix[off:off+4])
			}
			for y := 0; y < h; y++ {
				lo := clampInt(y-radius, 0, h-1)
				hi := clampInt(y+radius, 0, h-1)
				var pr, pg, pb, pa float64
				for iy := lo; iy <= hi; iy++ {
					weight := kernel[absint(y-iy)]
					i := iy * 4
					a := column[i+3]
					pr += float64(mul8(column[i], a)) * weight
					pg += float64(mul8(column[i+1], a)) * weight
					pb += float64(mul8(column[i+2], a)) * weight
					pa += float64(a) * weight
				}
				off := y*dst.Stride + x*4
				if pa <= 0 {
					dst.Pix[off], dst.Pix[off+1], dst.Pix[off+2], dst.Pix[off+3] = 0, 0, 0, 0
					continue
				}
				dst.Pix[off] = clamp(pr * 255 / pa)
				dst.Pix[off+1] = clamp(pg * 255 / pa)
				dst.Pix[off+2] = clamp(pb * 255 / pa)
				dst.Pix[off+3] = clamp(pa)
			}
		}
	})

	return dst, nil
}

// EdgeKernelKind selects a standard edge-detection kernel.
type EdgeKernelKind int

const (
	// EdgeSobel is the Sobel gradient-magnitude kernel.
	EdgeSobel EdgeKernelKind = iota
	// EdgeLaplacian is the 3x3 Laplacian kernel.
	EdgeLaplacian
)

// DetectEdgesChecked runs a 3x3 edge detector and returns the gradient
// magnitude image. A single-pixel image is valid and returns an all-zero
// (border clamped) result.
func DetectEdgesChecked(img image.Image, kind EdgeKernelKind) (*image.NRGBA, error) {
	bounds, err := validateImage(img)
	if err != nil {
		return nil, err
	}
	if kind != EdgeSobel && kind != EdgeLaplacian {
		return nil, ErrInvalidParameter
	}

	gray := Grayscale(img)
	w, h := bounds.Dx(), bounds.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))

	var kx, ky [9]float64
	switch kind {
	case EdgeSobel:
		kx = [9]float64{-1, 0, 1, -2, 0, 2, -1, 0, 1}
		ky = [9]float64{-1, -2, -1, 0, 0, 0, 1, 2, 1}
	case EdgeLaplacian:
		kx = [9]float64{0, 1, 0, 1, -4, 1, 0, 1, 0}
	}

	parallel(0, h, func(ys <-chan int) {
		for y := range ys {
			for x := 0; x < w; x++ {
				var gx, gy float64
				idx := 0
				for ky1 := -1; ky1 <= 1; ky1++ {
					for kx1 := -1; kx1 <= 1; kx1++ {
						ix := clampInt(x+kx1, 0, w-1)
						iy := clampInt(y+ky1, 0, h-1)
						v := float64(gray.Pix[iy*gray.Stride+ix*4])
						gx += v * kx[idx]
						gy += v * ky[idx]
						idx++
					}
				}
				mag := math.Abs(gx) + math.Abs(gy)
				if kind == EdgeLaplacian {
					mag = math.Abs(gx)
				}
				c := clamp(mag)
				off := y*dst.Stride + x*4
				dst.Pix[off], dst.Pix[off+1], dst.Pix[off+2] = c, c, c
				dst.Pix[off+3] = gray.Pix[y*gray.Stride+x*4+3]
			}
		}
	})
	return dst, nil
}
