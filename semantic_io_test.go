package imaging

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/gif"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func openFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "semantic", name))
	if err != nil {
		t.Fatalf("fixture %s: %v", name, err)
	}
	return data
}

func TestDecodeResourcePNGFixtures(t *testing.T) {
	for _, tc := range []struct {
		file         string
		wantW, wantH int
		wantFormat   Format
	}{
		{"odd_5x3.png", 5, 3, PNG},
		{"odd_3x5.png", 3, 5, PNG},
		{"alpha_4x4.png", 4, 4, PNG},
		{"channels_3x2.png", 3, 2, PNG},
		{"single_1x1.png", 1, 1, PNG},
		{"photo_7x5.jpg", 7, 5, JPEG},
	} {
		res, err := DecodeResource(bytes.NewReader(openFixture(t, tc.file)))
		if err != nil {
			t.Fatalf("%s: %v", tc.file, err)
		}
		if res.Format != tc.wantFormat {
			t.Fatalf("%s format = %v want %v", tc.file, res.Format, tc.wantFormat)
		}
		if len(res.Frames) != 1 {
			t.Fatalf("%s frames = %d want 1", tc.file, len(res.Frames))
		}
		if res.Width != tc.wantW || res.Height != tc.wantH {
			t.Fatalf("%s size = %dx%d want %dx%d", tc.file, res.Width, res.Height, tc.wantW, tc.wantH)
		}
		if res.Frames[0].Image.Rect.Dx() != tc.wantW || res.Frames[0].Image.Rect.Dy() != tc.wantH {
			t.Fatalf("%s frame size mismatch", tc.file)
		}
	}
}

func TestDecodeResourceChannelOrder(t *testing.T) {
	res, err := DecodeResource(bytes.NewReader(openFixture(t, "channels_3x2.png")))
	if err != nil {
		t.Fatal(err)
	}
	want := []color.NRGBA{
		{255, 0, 0, 255},
		{0, 255, 0, 255},
		{0, 0, 255, 255},
		{10, 20, 30, 128},
		{40, 50, 60, 64},
		{70, 80, 90, 0},
	}
	got := res.Frames[0].Image
	for i, w := range want {
		off := i * 4
		gotC := color.NRGBA{got.Pix[off], got.Pix[off+1], got.Pix[off+2], got.Pix[off+3]}
		if gotC != w {
			t.Fatalf("pixel %d: got %v want %v", i, gotC, w)
		}
	}
}

func TestDecodeResourceGIFFrameOrderAndMetadata(t *testing.T) {
	res, err := DecodeResource(bytes.NewReader(openFixture(t, "frames_4x3.gif")))
	if err != nil {
		t.Fatal(err)
	}
	if res.Format != GIF {
		t.Fatalf("format = %v want GIF", res.Format)
	}
	if len(res.Frames) != 3 {
		t.Fatalf("frames = %d want 3", len(res.Frames))
	}
	if res.Metadata.LoopCount != 2 {
		t.Fatalf("loop count = %d want 2", res.Metadata.LoopCount)
	}
	for i, f := range res.Frames {
		if f.Index != i {
			t.Fatalf("frame %d has index %d", i, f.Index)
		}
		if f.Delay != 10+i*5 {
			t.Fatalf("frame %d delay = %d want %d", i, f.Delay, 10+i*5)
		}
		if f.Image.Rect.Dx() != 4 || f.Image.Rect.Dy() != 3 {
			t.Fatalf("frame %d size: %v", i, f.Image.Rect)
		}
	}
	// Frames must be distinct and strictly increasing in red-channel value
	// (the palette quantization is lossy, so exact values are not asserted).
	first := res.Frames[0].Image.Pix[0]
	second := res.Frames[1].Image.Pix[0]
	third := res.Frames[2].Image.Pix[0]
	if !(first < second && second < third) {
		t.Fatalf("frame ordering wrong: R channels %d,%d,%d must be strictly increasing", first, second, third)
	}
}

func TestStreamGIFReleasesAndCloses(t *testing.T) {
	r := io.NopCloser(bytes.NewReader(openFixture(t, "frames_4x3.gif")))
	tracking := &closeTrackingReader{Reader: r}
	stream, err := OpenStreamGIF(tracking)
	if err != nil {
		t.Fatal(err)
	}
	var frames int
	for {
		f, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if f == nil || f.Image == nil {
			t.Fatal("nil frame delivered")
		}
		frames++
	}
	if frames != 3 {
		t.Fatalf("streamed frames = %d want 3", frames)
	}
	if !tracking.closed {
		t.Fatal("reader not closed after EOF")
	}
	// Close must be idempotent.
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	// Next after close returns EOF without panic.
	if _, err := stream.Next(); err != io.EOF {
		t.Fatalf("Next after close = %v want EOF", err)
	}
}

type closeTrackingReader struct {
	io.Reader
	closed bool
	mu     sync.Mutex
}

func (c *closeTrackingReader) Read(p []byte) (int, error) { return c.Reader.Read(p) }
func (c *closeTrackingReader) Close() error {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	return nil
}

func TestDecodeResourceErrors(t *testing.T) {
	var err error
	// Unknown format.
	_, err = DecodeResource(bytes.NewReader(openFixture(t, "unknown.bin")))
	if !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("unknown format: got %v want ErrUnsupportedFormat", err)
	}

	// Truncated stream must classify as ErrTruncated.
	_, err = DecodeResource(bytes.NewReader(openFixture(t, "truncated.png")))
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("truncated stream: got %v want ErrTruncated", err)
	}

	// Empty stream.
	_, err = DecodeResource(bytes.NewReader(nil))
	if !errors.Is(err, ErrTruncated) {
		t.Fatalf("empty stream: got %v", err)
	}

	// Nil reader safety is exercised through a failing reader.
	_, err = DecodeResource(errReader{})
	if err == nil {
		t.Fatal("failing reader expected error")
	}
}

type errReader struct{}

func (errReader) Read(p []byte) (int, error) { return 0, errors.New("read failure") }

func TestEncodeResourceRoundTrip(t *testing.T) {
	src, err := DecodeResource(bytes.NewReader(openFixture(t, "odd_5x3.png")))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := EncodeResource(&buf, src, PNGCompressionLevel(png.BestSpeed)); err != nil {
		t.Fatal(err)
	}
	back, err := DecodeResource(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if back.Width != 5 || back.Height != 3 {
		t.Fatalf("round trip size: %dx%d", back.Width, back.Height)
	}
	// PNG is lossless: pixels must match exactly.
	if !compareBytes(back.Frames[0].Image.Pix, src.Frames[0].Image.Pix, 0) {
		t.Fatal("PNG round trip altered pixels")
	}

	// JPEG and BMP/TIFF paths must not error.
	for _, f := range []Format{JPEG, BMP, TIFF} {
		var b bytes.Buffer
		if err := EncodeResource(&b, src, JPEGQuality(90)); err != nil {
			t.Fatalf("format %v: %v", f, err)
		}
		if b.Len() == 0 {
			t.Fatalf("format %v produced no bytes", f)
		}
	}

	// No frames.
	if err := EncodeResource(io.Discard, &Resource{}); !errors.Is(err, ErrNoFrames) {
		t.Fatalf("empty resource: got %v want ErrNoFrames", err)
	}
}

func TestEncodeResourceGIFSizingAndFrameData(t *testing.T) {
	res, err := DecodeResource(bytes.NewReader(openFixture(t, "frames_4x3.gif")))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := EncodeResource(&buf, res); err != nil {
		t.Fatal(err)
	}
	back, err := gif.DecodeAll(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Image) != 3 {
		t.Fatalf("encoded GIF frames = %d want 3", len(back.Image))
	}
	for i := range back.Delay {
		if back.Delay[i] != res.Frames[i].Delay {
			t.Fatalf("delay %d: %d want %d", i, back.Delay[i], res.Frames[i].Delay)
		}
	}

	// Inconsistent frame dimensions must fail.
	bad := &Resource{
		Format: GIF,
		Frames: []Frame{
			{Image: image.NewNRGBA(image.Rect(0, 0, 4, 3))},
			{Image: image.NewNRGBA(image.Rect(0, 0, 2, 2))},
		},
	}
	if err := EncodeResource(io.Discard, bad); !errors.Is(err, ErrFrameSize) {
		t.Fatalf("inconsistent frames: got %v want ErrFrameSize", err)
	}
}

func TestBatchProcessOrderingAndParallelism(t *testing.T) {
	const n = 12
	items := make([]BatchItem, n)
	fixture := openFixture(t, "odd_5x3.png")
	for i := 0; i < n; i++ {
		out := &lockedBuffer{}
		items[i] = BatchItem{
			Input:  bytes.NewReader(fixture),
			Output: out,
			Format: PNG,
			Transform: func(img image.Image) (*image.NRGBA, error) {
				return ResizeChecked(img, 4, 2, Box)
			},
		}
	}
	results, err := BatchProcess(context.Background(), items)
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range results {
		if r.Index != i {
			t.Fatalf("result %d has index %d", i, r.Index)
		}
		if r.Err != nil {
			t.Fatalf("item %d failed: %v", i, r.Err)
		}
		buf := items[i].Output.(*lockedBuffer)
		if buf.Len() == 0 {
			t.Fatalf("item %d wrote nothing", i)
		}
	}

	// Every output must decode to the transform's deterministic dimensions.
	for i, item := range items {
		res, err := DecodeResource(bytes.NewReader(item.Output.(*lockedBuffer).Bytes()))
		if err != nil {
			t.Fatalf("item %d decode: %v", i, err)
		}
		if res.Width != 4 || res.Height != 2 {
			t.Fatalf("item %d size %dx%d", i, res.Width, res.Height)
		}
	}
}

func TestBatchProcessDeterministicRepeatedRuns(t *testing.T) {
	fixture := openFixture(t, "channels_3x2.png")
	run := func() [][]byte {
		items := make([]BatchItem, 6)
		for i := range items {
			out := &lockedBuffer{}
			items[i] = BatchItem{
				Input:  bytes.NewReader(fixture),
				Output: out,
				Format: PNG,
			}
		}
		results, err := BatchProcess(context.Background(), items)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range results {
			if r.Err != nil {
				t.Fatal(r.Err)
			}
		}
		out := make([][]byte, len(items))
		for i, item := range items {
			out[i] = append([]byte(nil), item.Output.(*lockedBuffer).Bytes()...)
		}
		return out
	}
	first := run()
	second := run()
	for i := range first {
		if !bytes.Equal(first[i], second[i]) {
			t.Fatalf("item %d output not deterministic across batches", i)
		}
	}
}

func TestBatchProcessCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already canceled
	items := make([]BatchItem, 4)
	fixture := openFixture(t, "odd_5x3.png")
	for i := range items {
		items[i] = BatchItem{
			Input:  bytes.NewReader(fixture),
			Output: &lockedBuffer{},
			Format: PNG,
		}
	}
	results, err := BatchProcess(ctx, items)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v want context.Canceled", err)
	}
	for _, r := range results {
		if r.Err != nil && !errors.Is(r.Err, context.Canceled) {
			t.Fatalf("unexpected per-item error: %v", r.Err)
		}
	}
}

func TestBatchProcessPerItemFailures(t *testing.T) {
	good := openFixture(t, "odd_5x3.png")
	items := []BatchItem{
		{Input: bytes.NewReader(good), Output: &lockedBuffer{}, Format: PNG},
		{Input: bytes.NewReader(openFixture(t, "truncated.png")), Output: &lockedBuffer{}, Format: PNG},
		{Input: bytes.NewReader(openFixture(t, "unknown.bin")), Output: &lockedBuffer{}, Format: PNG},
		{Input: nil, Output: &lockedBuffer{}, Format: PNG},
	}
	results, err := BatchProcess(context.Background(), items)
	if err != nil {
		t.Fatalf("batch must report per-item errors, not abort: %v", err)
	}
	if results[0].Err != nil {
		t.Fatalf("good item failed: %v", results[0].Err)
	}
	for i := 1; i < len(items); i++ {
		if results[i].Err == nil {
			t.Fatalf("bad item %d unexpectedly succeeded", i)
		}
	}
}

func TestSaveAtomicRollback(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.png")
	original := []byte("pre-existing destination content")
	if err := os.WriteFile(dst, original, 0o644); err != nil {
		t.Fatal(err)
	}

	// A writer failure must leave the original file intact.
	img := nrgba(2, 2)
	fillNRGBA(img, color.NRGBA{1, 2, 3, 255})
	// Force failure by targeting an unwritable directory.
	badPath := filepath.Join(dir, "no-such-dir", "x.png")
	if err := SaveAtomic(img, badPath); err == nil {
		t.Fatal("expected SaveAtomic to fail")
	}
	// Original destination untouched.
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, original) {
		t.Fatal("SaveAtomic failure overwrote existing file")
	}

	// Successful path replaces the destination atomically.
	if err := SaveAtomic(img, dst, PNGCompressionLevel(png.BestSpeed)); err != nil {
		t.Fatal(err)
	}
	res, err := DecodeResource(mustOpenReader(t, dst))
	if err != nil {
		t.Fatal(err)
	}
	if res.Width != 2 || res.Height != 2 {
		t.Fatalf("atomic save size: %dx%d", res.Width, res.Height)
	}
	// No temp files left behind.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".imaging-") {
			t.Fatalf("temporary file leaked: %s", e.Name())
		}
	}
}

func TestRetryEncode(t *testing.T) {
	img := nrgba(2, 2)
	fillNRGBA(img, color.NRGBA{10, 20, 30, 255})
	var attempts int
	shouldFail := true
	// Each attempt must receive a fresh writer; the first one fails and every
	// subsequent writer succeeds.
	err := RetryEncode(context.Background(), img, PNG, 3, func() (io.WriteCloser, error) {
		attempts++
		fail := shouldFail
		if shouldFail {
			shouldFail = false
		}
		return &retryWriteCloser{fail: func() bool { return fail }}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d want 2", attempts)
	}

	// Exhausted retries return the last error.
	attempts = 0
	err = RetryEncode(context.Background(), img, PNG, 2, func() (io.WriteCloser, error) {
		return &retryWriteCloser{fail: func() bool { return true }}, nil
	})
	if err == nil {
		t.Fatal("expected retry exhaustion error")
	}

	// Cancellation aborts without further attempts.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := RetryEncode(ctx, img, PNG, 5, func() (io.WriteCloser, error) {
		return &retryWriteCloser{}, nil
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("retry cancel: got %v", err)
	}

	if err := RetryEncode(context.Background(), nil, PNG, 1, func() (io.WriteCloser, error) {
		return &retryWriteCloser{}, nil
	}); !errors.Is(err, ErrNilImage) {
		t.Fatalf("nil image retry: %v", err)
	}
}

type retryWriteCloser struct {
	buf  bytes.Buffer
	fail func() bool
}

func (r *retryWriteCloser) Write(p []byte) (int, error) {
	if r.fail != nil && r.fail() {
		return 0, errors.New("simulated write failure")
	}
	return r.buf.Write(p)
}
func (r *retryWriteCloser) Close() error { return nil }

func mustOpenReader(t *testing.T, path string) io.Reader {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.NewReader(data)
}

// lockedBuffer is a thread-safe bytes.Buffer used as batch output sink.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}
func (b *lockedBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Len()
}
func (b *lockedBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Bytes()
}
