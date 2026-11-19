// Command gensemanticfixtures generates the offline fixtures used by the
// semantic processing path tests. The generator lives outside the package
// testdata directory and writes deterministic, fully reproducible files:
// rerunning it produces byte-identical output.
//
// Usage:
//
//	go run ./cmd/gensemanticfixtures -out testdata/semantic
package main

import (
	"bytes"
	"flag"
	"image"
	"image/color"
	"image/color/palette"
	"image/gif"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
)

func main() {
	out := flag.String("out", "testdata/semantic", "output directory")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		panic(err)
	}

	must(writePNG(filepath.Join(*out, "odd_5x3.png"), patternImage(5, 3)))
	must(writePNG(filepath.Join(*out, "odd_3x5.png"), patternImage(3, 5)))
	must(writePNG(filepath.Join(*out, "alpha_4x4.png"), alphaImage(4, 4)))
	must(writePNG(filepath.Join(*out, "channels_3x2.png"), channelImage()))
	must(writePNG(filepath.Join(*out, "single_1x1.png"), patternImage(1, 1)))

	var jpegBuf bytes.Buffer
	if err := jpeg.Encode(&jpegBuf, patternImage(7, 5), &jpeg.Options{Quality: 90}); err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(*out, "photo_7x5.jpg"), jpegBuf.Bytes(), 0o644); err != nil {
		panic(err)
	}

	// Three-frame animated GIF with distinct per-frame pixels.
	frames := make([]*image.Paletted, 3)
	delays := make([]int, 3)
	for f := 0; f < 3; f++ {
		img := image.NewPaletted(image.Rect(0, 0, 4, 3), palette.Plan9[:256])
		for y := 0; y < 3; y++ {
			for x := 0; x < 4; x++ {
				img.Set(x, y, color.NRGBA{
					R: uint8((f*60 + x*30) & 0xff),
					G: uint8((y * 70) & 0xff),
					B: uint8((f*40 + x*20 + y*10) & 0xff),
					A: 0xff,
				})
			}
		}
		frames[f] = img
		delays[f] = 10 + f*5
	}
	gifFile, err := os.Create(filepath.Join(*out, "frames_4x3.gif"))
	if err != nil {
		panic(err)
	}
	if err := gif.EncodeAll(gifFile, &gif.GIF{Image: frames, Delay: delays, LoopCount: 2}); err != nil {
		panic(err)
	}
	if err := gifFile.Close(); err != nil {
		panic(err)
	}

	// Truncated PNG: valid signature plus a cut-off IHDR.
	good := pngBytes(patternImage(4, 4))
	if err := os.WriteFile(filepath.Join(*out, "truncated.png"), good[:len(good)/2], 0o644); err != nil {
		panic(err)
	}
	// Unknown format content.
	if err := os.WriteFile(filepath.Join(*out, "unknown.bin"), []byte("definitely not an image\xff\xd8nope"), 0o644); err != nil {
		panic(err)
	}
}

// patternImage builds a width x height image where every pixel value is a
// deterministic function of its coordinates, with mixed alpha values.
func patternImage(width, height int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			off := y*img.Stride + x*4
			img.Pix[off] = uint8((x*37 + 11) & 0xff)
			img.Pix[off+1] = uint8((y*53 + 29) & 0xff)
			img.Pix[off+2] = uint8((x*17 + y*23 + 7) & 0xff)
			img.Pix[off+3] = uint8(64 + ((x+y)*191)/(width+height))
		}
	}
	return img
}

func alphaImage(width, height int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			off := y*img.Stride + x*4
			img.Pix[off] = 200
			img.Pix[off+1] = 100
			img.Pix[off+2] = 50
			// Left half fully transparent, right half partially/fully opaque.
			if x < width/2 {
				img.Pix[off+3] = 0
			} else {
				img.Pix[off+3] = uint8(80 + x*20)
			}
		}
	}
	return img
}

func channelImage() *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	colors := []color.NRGBA{
		{255, 0, 0, 255},
		{0, 255, 0, 255},
		{0, 0, 255, 255},
		{10, 20, 30, 128},
		{40, 50, 60, 64},
		{70, 80, 90, 0},
	}
	for i, c := range colors {
		off := i * 4
		img.Pix[off], img.Pix[off+1], img.Pix[off+2], img.Pix[off+3] = c.R, c.G, c.B, c.A
	}
	return img
}

func writePNG(path string, img image.Image) error {
	return os.WriteFile(path, pngBytes(img), 0o644)
}

func pngBytes(img image.Image) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
