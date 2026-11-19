package imaging

import (
	"errors"
	"image"
	"math"
	"reflect"
	"sync"
	"unsafe"
)

// Semantic processing path.
//
// The functions in semantic*.go complement the legacy entry points (Resize,
// Crop, Rotate, Adjust*, Encode, ...) which keep their original signatures and
// behavior. Every semantic function validates its inputs before touching any
// pixel buffer, returns a stable sentinel error on failure and never mutates
// the caller's source image. Rounding, sampling coordinates and color pipeline
// ordering are fixed and documented so that repeated runs and an explicit
// reference implementation produce byte-identical results.

// Sentinel errors returned by the semantic processing path.
var (
	// ErrNilImage is returned when a nil image (or a nil typed image pointer)
	// is passed to a semantic function.
	ErrNilImage = errors.New("imaging: nil image")

	// ErrEmptyImage is returned when the image has zero width or zero height.
	ErrEmptyImage = errors.New("imaging: empty image")

	// ErrInvalidSize is returned when a requested width, height or rectangle
	// is not valid for the operation.
	ErrInvalidSize = errors.New("imaging: invalid image dimensions")

	// ErrInvalidParameter is returned when a parameter (percentage, gamma,
	// color, ...) is outside its documented range.
	ErrInvalidParameter = errors.New("imaging: invalid parameter")

	// ErrAliasedImage is returned when a destination buffer shares pixel
	// storage with the source image and writing through it would corrupt the
	// source before processing finished.
	ErrAliasedImage = errors.New("imaging: destination aliases source image")

	// ErrRadiusTooLarge is returned when a convolution or blur radius is not
	// strictly smaller than the image extent, or when it exceeds the hard
	// MaxKernelRadius limit.
	ErrRadiusTooLarge = errors.New("imaging: kernel radius too large")

	// ErrNoFrames is returned when an image resource is expected to contain at
	// least one frame but contains none.
	ErrNoFrames = errors.New("imaging: resource contains no frames")

	// ErrTruncated wraps decoding failures caused by an incomplete or corrupt
	// image stream. It matches errors.Is.
	ErrTruncated = errors.New("imaging: truncated image stream")

	// ErrFrameSize is returned when multi-frame encoding is requested for
	// frames with inconsistent dimensions.
	ErrFrameSize = errors.New("imaging: inconsistent frame dimensions")

	// ErrSizeMismatch is returned when a caller-provided destination buffer
	// does not match the produced image size.
	ErrSizeMismatch = errors.New("imaging: destination size mismatch")
)

// MaxKernelRadius is the hard upper bound for any convolution or blur kernel
// radius accepted by the semantic processing path. The bound keeps buffer
// allocations bounded even when a caller passes an extreme parameter such as
// math.MaxFloat64 as the blur sigma.
const MaxKernelRadius = 1024

// validateImage checks that img is non-nil and has a positive width and
// height. It returns the image bounds on success.
func validateImage(img image.Image) (image.Rectangle, error) {
	if img == nil {
		return image.Rectangle{}, ErrNilImage
	}
	v := reflect.ValueOf(img)
	if v.Kind() == reflect.Ptr && v.IsNil() {
		return image.Rectangle{}, ErrNilImage
	}
	b := img.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		return b, ErrEmptyImage
	}
	return b, nil
}

// validateSize checks a destination width/height pair.
func validateSize(width, height int) error {
	if width <= 0 || height <= 0 {
		return ErrInvalidSize
	}
	if width > MaxImageDimension || height > MaxImageDimension {
		return ErrInvalidSize
	}
	return nil
}

// MaxImageDimension is the largest accepted image edge length. It protects
// against integer overflow in stride calculations (4*w*h) while staying well
// above any real-world image.
const MaxImageDimension = 1 << 24

// validateRadius checks a kernel radius against the global kernel limit.
// Radius is allowed to reach the image extent: edge sampling clamps to the
// border, which is well-defined even for a single-pixel image.
func validateRadius(radius, w, h int) error {
	if radius < 0 {
		return ErrInvalidParameter
	}
	if radius > MaxKernelRadius {
		return ErrRadiusTooLarge
	}
	return nil
}

// normalizeAngle maps an arbitrary rotation angle in degrees to the
// half-open interval [0, 360). Negative angles are supported. It returns
// ErrInvalidParameter for NaN or Inf values.
func normalizeAngle(angle float64) (float64, error) {
	if math.IsNaN(angle) || math.IsInf(angle, 0) {
		return 0, ErrInvalidParameter
	}
	return angle - math.Floor(angle/360.0)*360.0, nil
}

// nearestSourceIndex is the single fixed nearest-neighbor sampling rule used
// by every nearest-neighbor code path. Pixel centers sit at integer + 0.5, so
// the source coordinate is (dstIndex+0.5)*srcSize/dstSize and the selected
// index is its floor (truncation toward zero for the always non-negative
// coordinate), clamped to the source extent.
func nearestSourceIndex(dstIndex, dstSize, srcSize int) int {
	coord := (float64(dstIndex) + 0.5) * float64(srcSize) / float64(dstSize)
	i := int(math.Floor(coord))
	if i < 0 {
		return 0
	}
	if i >= srcSize {
		return srcSize - 1
	}
	return i
}

// Aliased reports whether dst shares underlying pixel storage with src.
// Concrete image types carrying a Pix slice are inspected through reflect so
// that all standard library image kinds (NRGBA, RGBA, Gray, Paletted, ...) are
// covered. A true result means that writing into dst could corrupt src.
func Aliased(src image.Image, dst *image.NRGBA) bool {
	if src == nil || dst == nil || len(dst.Pix) == 0 {
		return false
	}
	srcBase, srcLen, ok := pixMemoryRange(src)
	if !ok || srcLen == 0 {
		return false
	}
	dstBase := unsafe.Pointer(&dst.Pix[0])
	dstLen := len(dst.Pix)
	return rangesOverlap(srcBase, uintptr(srcLen), dstBase, uintptr(dstLen))
}

// pixMemoryRange returns the base address and byte length of the Pix backing
// array of a standard library image type.
func pixMemoryRange(img image.Image) (unsafe.Pointer, uintptr, bool) {
	v := reflect.ValueOf(img)
	if v.Kind() != reflect.Ptr || v.IsNil() {
		return nil, 0, false
	}
	pix := v.Elem().FieldByName("Pix")
	if !pix.IsValid() || pix.Kind() != reflect.Slice || pix.Len() == 0 {
		return nil, 0, false
	}
	return unsafe.Pointer(pix.Index(0).Addr().Pointer()), uintptr(pix.Len()), true
}

func rangesOverlap(base1 unsafe.Pointer, len1 uintptr, base2 unsafe.Pointer, len2 uintptr) bool {
	s1 := uintptr(base1)
	s2 := uintptr(base2)
	e1 := s1 + len1
	e2 := s2 + len2
	return s1 < e2 && s2 < e1
}

// CheckedTransform is an image operation that validates its inputs and reports
// failures instead of silently returning an empty image.
type CheckedTransform func(img image.Image) (*image.NRGBA, error)

// TransformInto runs fn on src and writes the result into the caller-provided
// dst. It fails with ErrAliasedImage if dst shares pixel storage with src, and
// with ErrSizeMismatch if dst cannot hold the result. The result is computed
// in a fresh buffer first, so any failure leaves dst (which may be the
// caller's original image) byte-for-byte untouched.
func TransformInto(dst *image.NRGBA, src image.Image, fn CheckedTransform) error {
	if dst == nil {
		return ErrNilImage
	}
	if _, err := validateImage(src); err != nil {
		return err
	}
	if Aliased(src, dst) {
		return ErrAliasedImage
	}
	out, err := fn(src)
	if err != nil {
		return err
	}
	if out.Rect.Dx() != dst.Rect.Dx() || out.Rect.Dy() != dst.Rect.Dy() {
		return ErrSizeMismatch
	}
	copy(dst.Pix, out.Pix)
	return nil
}

// BufferPool hands out reusable byte buffers. It is backed by sync.Pool and is
// safe for concurrent use: every Get returns a slice owned exclusively by the
// caller until it is Put back. Buffers that grew beyond poolMaxBytes are
// dropped on Put instead of being retained.
type BufferPool struct {
	once sync.Once
	pool sync.Pool
}

const poolMaxBytes = 64 << 20 // 64 MiB per retained buffer

// Get returns a byte buffer of at least size bytes.
func (p *BufferPool) Get(size int) []uint8 {
	p.once.Do(func() {
		p.pool.New = func() interface{} {
			b := make([]uint8, 0)
			return &b
		}
	})
	if size < 0 {
		size = 0
	}
	bp := p.pool.Get().(*[]uint8)
	if cap(*bp) < size {
		*bp = make([]uint8, size)
	} else {
		*bp = (*bp)[:size]
	}
	return *bp
}

// Put releases a buffer back to the pool. The caller must not retain or access
// the buffer after Put.
func (p *BufferPool) Put(buf []uint8) {
	if cap(buf) == 0 || cap(buf) > poolMaxBytes {
		return
	}
	p.once.Do(func() {
		p.pool.New = func() interface{} {
			b := make([]uint8, 0)
			return &b
		}
	})
	b := buf[:cap(buf)]
	p.pool.Put(&b)
}

// sharedBufferPool is used by semantic processing functions for per-goroutine
// scan-line scratch space.
var sharedBufferPool BufferPool
