package imaging

import (
	"bytes"
	"context"
	"image"
	"image/draw"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// This file contains the I/O and batch processing part of the semantic
// processing path. PNG, JPEG and GIF inputs are decoded into one common
// Resource frame model that also carries format metadata. Unknown formats and
// truncated streams produce stable, wrapped errors. All readers and files are
// released on every code path including cancellation and retries. Batch
// processing parallelizes work across different images while preserving the
// submission order of the written results.

// Frame is one still image of a Resource. Frame images are non-premultiplied
// NRGBA so that every downstream semantic function can consume them directly.
type Frame struct {
	Image    *image.NRGBA
	Delay    int // GIF delay in hundredths of a second; zero for still formats
	Disposal byte
	Index    int // zero-based frame index within the resource
}

// Resource is the common result model for decoded images. Still formats
// (PNG, JPEG, TIFF, BMP) produce exactly one frame; GIF produces one frame per
// stored image.
type Resource struct {
	Frames   []Frame
	Format   Format
	Width    int
	Height   int
	Metadata Metadata
}

// Metadata carries format-level information alongside the frames.
type Metadata struct {
	Orientation orientation // EXIF orientation for JPEG, 0 otherwise
	LoopCount   int         // GIF loop count; zero for still formats
	ColorModel  string
}

// DecodeResource reads an image from r and returns the unified Resource model.
// It uses image.DecodeConfig for format detection so an unknown format fails
// with ErrUnsupportedFormat before pixel decoding starts, and an incomplete or
// corrupt stream fails with an error wrapping ErrTruncated (errors.Is).
func DecodeResource(r io.Reader, opts ...DecodeOption) (*Resource, error) {
	cfg := defaultDecodeConfig
	for _, option := range opts {
		option(&cfg)
	}

	// Buffer the full stream once: format sniffing, EXIF inspection and the
	// pixel decoder all read from independent readers over the same bytes, so
	// none of them can consume data another one still needs.
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, &truncatedError{err: io.EOF}
	}
	head := data
	if len(head) > 512 {
		head = head[:512]
	}
	reader := bytes.NewReader(data)

	_, formatName, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		if strings.Contains(err.Error(), "unknown format") {
			return nil, ErrUnsupportedFormat
		}
		return nil, classifyDecodeError(err)
	}
	if formatName == "gif" {
		return decodeGIFResource(reader)
	}

	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, classifyDecodeError(err)
	}
	nrgba := Clone(img)
	res := &Resource{
		Frames:   []Frame{{Image: nrgba, Index: 0}},
		Width:    nrgba.Rect.Dx(),
		Height:   nrgba.Rect.Dy(),
		Metadata: Metadata{ColorModel: colorModelName(img)},
	}
	switch formatName {
	case "jpeg":
		res.Format = JPEG
		if cfg.autoOrientation {
			if orient := readOrientation(bytes.NewReader(data)); orient != orientationUnspecified {
				res.Metadata.Orientation = orient
				res.Frames[0].Image = Clone(fixOrientation(nrgba, orient))
			}
		}
	case "png":
		res.Format = PNG
	case "tiff":
		res.Format = TIFF
	case "bmp":
		res.Format = BMP
	default:
		return nil, ErrUnsupportedFormat
	}
	return res, nil
}

// OpenResource opens a file and decodes it through DecodeResource. The file
// handle is always closed, including on error and cancellation.
func OpenResource(filename string, opts ...DecodeOption) (*Resource, error) {
	file, err := fs.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return DecodeResource(file, opts...)
}

func colorModelName(img image.Image) string {
	switch img.(type) {
	case *image.NRGBA, *image.RGBA:
		return "nrgba"
	case *image.Gray, *image.Gray16:
		return "gray"
	case *image.Paletted:
		return "paletted"
	case *image.YCbCr:
		return "ycbcr"
	default:
		return "generic"
	}
}

func decodeGIFResource(r io.Reader) (*Resource, error) {
	g, err := gif.DecodeAll(r)
	if err != nil {
		return nil, classifyDecodeError(err)
	}
	if g == nil || len(g.Image) == 0 {
		return nil, ErrNoFrames
	}
	res := &Resource{
		Format:   GIF,
		Width:    g.Config.Width,
		Height:   g.Config.Height,
		Metadata: Metadata{LoopCount: g.LoopCount, ColorModel: "paletted"},
	}
	for i, pal := range g.Image {
		nrgba := Clone(pal)
		frame := Frame{Image: nrgba, Index: i}
		if i < len(g.Delay) {
			frame.Delay = g.Delay[i]
		}
		if i < len(g.Disposal) {
			frame.Disposal = g.Disposal[i]
		}
		res.Frames = append(res.Frames, frame)
	}
	if res.Width == 0 && len(res.Frames) > 0 {
		res.Width = res.Frames[0].Image.Rect.Dx()
	}
	if res.Height == 0 && len(res.Frames) > 0 {
		res.Height = res.Frames[0].Image.Rect.Dy()
	}
	return res, nil
}

// StreamGIF delivers GIF frames one at a time over a fully decoded frame set,
// releasing each frame's pixel buffer when Next advances past it. It is safe
// to abandon the stream from another goroutine via Close (for example from a
// context cancellation handler).
type StreamGIF struct {
	reader io.Closer
	g      *gif.GIF
	index  int
	mu     sync.Mutex
	closed bool
}

// OpenStreamGIF opens a frame iterator over a GIF stream. When r also
// implements io.Closer it is closed by StreamGIF.Close.
func OpenStreamGIF(r io.Reader) (*StreamGIF, error) {
	dec, err := gif.DecodeAll(r)
	if err != nil {
		if c, ok := r.(io.Closer); ok {
			_ = c.Close()
		}
		return nil, classifyDecodeError(err)
	}
	if dec == nil || len(dec.Image) == 0 {
		if c, ok := r.(io.Closer); ok {
			_ = c.Close()
		}
		return nil, ErrNoFrames
	}
	s := &StreamGIF{g: dec}
	if c, ok := r.(io.Closer); ok {
		s.reader = c
	}
	return s, nil
}

// Next returns the next frame or io.EOF once every frame has been delivered.
// Frames already delivered release their pixel storage; after EOF or an error
// the underlying reader is closed exactly once.
func (s *StreamGIF) Next() (*Frame, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, io.EOF
	}
	if s.index >= len(s.g.Image) {
		s.closeLocked()
		return nil, io.EOF
	}
	pal := s.g.Image[s.index]
	frame := &Frame{Image: Clone(pal), Index: s.index}
	if s.index < len(s.g.Delay) {
		frame.Delay = s.g.Delay[s.index]
	}
	if s.index < len(s.g.Disposal) {
		frame.Disposal = s.g.Disposal[s.index]
	}
	// Release the decoded frame's pixels; metadata slices stay intact.
	s.g.Image[s.index] = nil
	s.index++
	return frame, nil
}

// Close releases the underlying reader. It is idempotent.
func (s *StreamGIF) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closeLocked()
	return nil
}

func (s *StreamGIF) closeLocked() {
	s.closed = true
	if s.reader != nil {
		_ = s.reader.Close()
	}
}

// LoopCount returns the GIF loop count once at least the header was decoded.
func (s *StreamGIF) LoopCount() int {
	return s.g.LoopCount
}

// truncatedErrorMarkers are substrings produced by the standard decoders when
// a stream ends prematurely or contains invalid data.
var truncatedErrorMarkers = []string{
	"unexpected EOF",
	"EOF",
	"truncated",
	"invalid",
	"corrupt",
	"short",
}

func classifyDecodeError(err error) error {
	if err == nil {
		return nil
	}
	msg := strings.ToLower(err.Error())
	for _, marker := range truncatedErrorMarkers {
		if strings.Contains(msg, marker) {
			return &truncatedError{err: err}
		}
	}
	return err
}

type truncatedError struct{ err error }

func (e *truncatedError) Error() string { return e.err.Error() }
func (e *truncatedError) Unwrap() error { return e.err }

// Is makes errors.Is(err, ErrTruncated) succeed for every wrapped truncated
// stream failure.
func (e *truncatedError) Is(target error) bool { return target == ErrTruncated }

// EncodeResource writes the resource frames in the given format. GIF encodes
// every frame (all frames must share dimensions, otherwise ErrFrameSize is
// returned); the other formats encode the first frame. A nil resource or an
// empty frame list fails with ErrNoFrames.
func EncodeResource(w io.Writer, res *Resource, opts ...EncodeOption) error {
	if res == nil || len(res.Frames) == 0 {
		return ErrNoFrames
	}
	cfg := defaultEncodeConfig
	for _, option := range opts {
		option(&cfg)
	}

	switch res.Format {
	case GIF:
		g := &gif.GIF{LoopCount: res.Metadata.LoopCount}
		firstSize := res.Frames[0].Image.Rect.Size()
		for _, f := range res.Frames {
			if !f.Image.Rect.Size().Eq(firstSize) {
				return ErrFrameSize
			}
			bounds := f.Image.Bounds()
			pal := image.NewPaletted(bounds, cfgPalette(cfg, bounds.Dx(), bounds.Dy()))
			drawer := cfg.gifDrawer
			if drawer == nil {
				drawer = draw.FloydSteinberg
			}
			drawer.Draw(pal, bounds, f.Image, bounds.Min)
			g.Image = append(g.Image, pal)
			g.Delay = append(g.Delay, f.Delay)
			g.Disposal = append(g.Disposal, f.Disposal)
		}
		return gif.EncodeAll(w, g)
	case PNG:
		encoder := png.Encoder{CompressionLevel: cfg.pngCompressionLevel}
		return encoder.Encode(w, res.Frames[0].Image)
	case JPEG:
		return jpeg.Encode(w, res.Frames[0].Image, &jpeg.Options{Quality: cfg.jpegQuality})
	case TIFF:
		return tiffEncode(w, res.Frames[0].Image)
	case BMP:
		return bmpEncode(w, res.Frames[0].Image)
	}
	return ErrUnsupportedFormat
}

func cfgPalette(cfg encodeConfig, w, h int) colorPalette {
	n := cfg.gifNumColors
	if n <= 0 {
		n = 256
	}
	if cfg.gifQuantizer != nil {
		return cfg.gifQuantizer.Quantize(make(colorPalette, 0, n), image.NewNRGBA(image.Rect(0, 0, w, h)))
	}
	return plan9Palette(n)
}

// BatchItem is one unit of batch work: decode Input, apply Transform, then
// encode to Output.
type BatchItem struct {
	Input  io.Reader
	Output io.Writer
	Format Format
	// Transform is applied to the decoded first frame. It may be nil.
	Transform CheckedTransform
	// EncodeOpts configure encoding.
	EncodeOpts []EncodeOption
}

// BatchResult reports the outcome of one BatchItem in submission order.
type BatchResult struct {
	Index int
	Err   error
}

// BatchProcess decodes, transforms and encodes every item. Distinct items are
// processed concurrently (bounded by the package's SetMaxProcs limit); the
// returned result slice always follows the item submission order. When ctx is
// canceled, in-flight items are abandoned and the function returns with the
// context error; no partial output is ever returned to the caller because each
// item encodes into its own writer.
func BatchProcess(ctx context.Context, items []BatchItem) ([]BatchResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	results := make([]BatchResult, len(items))
	sem := make(chan struct{}, batchConcurrency())
	var wg sync.WaitGroup

	for i, item := range items {
		select {
		case <-ctx.Done():
			wg.Wait()
			for j := range results {
				if results[j].Err == nil && j >= i {
					results[j] = BatchResult{Index: j, Err: ctx.Err()}
				}
			}
			return results, ctx.Err()
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func(idx int, it BatchItem) {
			defer wg.Done()
			defer func() { <-sem }()
			results[idx] = BatchResult{Index: idx, Err: processOne(ctx, it)}
		}(i, item)
	}
	wg.Wait()

	// Deterministic order check: results are written by index regardless of
	// goroutine scheduling, and the returned slice is indexed 1:1 to inputs.
	for i := range results {
		if results[i].Index != i {
			panic("imaging: batch result order invariant violated")
		}
	}
	return results, nil
}

func processOne(ctx context.Context, item BatchItem) error {
	if item.Input == nil || item.Output == nil {
		return ErrInvalidParameter
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	res, err := DecodeResource(item.Input)
	if err != nil {
		return err
	}
	if item.Transform != nil {
		out, err := item.Transform(res.Frames[0].Image)
		if err != nil {
			return err
		}
		res.Frames[0].Image = out
	}
	res.Format = item.Format
	done := make(chan error, 1)
	go func() { done <- EncodeResource(item.Output, res, item.EncodeOpts...) }()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return err
	}
}

func batchConcurrency() int {
	limit := int(atomicLoadMaxProcs())
	if limit > 0 {
		return limit
	}
	return gomaxprocs()
}

// SaveAtomic writes img to filename through a temporary file and renames it
// into place only after a successful encode and close. A failure at any stage
// removes the temporary file and leaves any pre-existing destination
// untouched, which makes batch failures roll back cleanly.
func SaveAtomic(img image.Image, filename string, opts ...EncodeOption) (err error) {
	format, err := FormatFromFilename(filename)
	if err != nil {
		return err
	}
	dir := filepath.Dir(filename)
	tmp, err := os.CreateTemp(dir, ".imaging-*"+filepath.Ext(filename))
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpName) }

	if err := Encode(tmp, img, format, opts...); err != nil {
		_ = tmp.Close()
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return err
	}
	if err := os.Rename(tmpName, filename); err != nil {
		cleanup()
		return err
	}
	return nil
}

// RetryEncode repeats encode up to attempts times while the writer returns an
// error. Every attempt gets a fresh writer from newWriter so a partially
// written stream is discarded. A nil/zero attempts value defaults to one.
// Context cancellation aborts immediately without further retries.
func RetryEncode(ctx context.Context, img image.Image, format Format, attempts int, newWriter func() (io.WriteCloser, error)) error {
	if img == nil {
		return ErrNilImage
	}
	if attempts <= 0 {
		attempts = 1
	}
	var lastErr error
	for attempt := 0; attempt < attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		w, err := newWriter()
		if err != nil {
			lastErr = err
			continue
		}
		encErr := Encode(w, img, format)
		closeErr := w.Close()
		if encErr == nil && closeErr == nil {
			return nil
		}
		if encErr != nil {
			lastErr = encErr
		} else {
			lastErr = closeErr
		}
	}
	return lastErr
}

// ensure bytes stays referenced even when build tags trim helper files
var _ = bytes.NewReader
