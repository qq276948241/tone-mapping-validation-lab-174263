package imaging

import (
	"bytes"
	"image"
	"testing"
)

func benchImage(b *testing.B, w, h int) *image.NRGBA {
	b.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = uint8((i*29 + 7) & 0xff)
	}
	return img
}

func BenchmarkResizeNearestChecked(b *testing.B) {
	src := benchImage(b, 640, 480)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ResizeChecked(src, 320, 240, NearestNeighbor); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkResizeLanczosChecked(b *testing.B) {
	src := benchImage(b, 640, 480)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ResizeChecked(src, 320, 240, Lanczos); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRotateChecked(b *testing.B) {
	src := benchImage(b, 480, 360)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RotateChecked(src, 37, opaqueBlack()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkColorPipeline(b *testing.B) {
	src := benchImage(b, 640, 480)
	adj := ColorAdjustments{Brightness: 8, Contrast: 12, Gamma: 1.3, Saturation: 18}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyColorPipeline(src, adj); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolve3x3Checked(b *testing.B) {
	src := benchImage(b, 480, 360)
	kernel := [9]float64{-1, -1, 0, -1, 1, 1, 0, 1, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Convolve3x3Checked(src, kernel, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBlurChecked(b *testing.B) {
	src := benchImage(b, 480, 360)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := BlurChecked(src, 2.0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPremultiplyRoundTrip(b *testing.B) {
	src := benchImage(b, 480, 360)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prem, err := Premultiply(src)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := Unpremultiply(prem); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHistogramChecked(b *testing.B) {
	src := benchImage(b, 480, 360)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := HistogramChecked(src); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeResourcePNG(b *testing.B) {
	src := benchImage(b, 320, 240)
	var buf bytes.Buffer
	if err := Encode(&buf, src, PNG); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeResource(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBatchProcess(b *testing.B) {
	src := benchImage(b, 160, 120)
	var buf bytes.Buffer
	if err := Encode(&buf, src, PNG); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		items := make([]BatchItem, 8)
		for j := range items {
			items[j] = BatchItem{
				Input:  bytes.NewReader(data),
				Output: &bytes.Buffer{},
				Format: PNG,
			}
		}
		if _, err := BatchProcess(batchBenchContext(), items); err != nil {
			b.Fatal(err)
		}
	}
}
