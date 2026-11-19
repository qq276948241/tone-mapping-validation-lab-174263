package imaging

import (
	"os"
	"path/filepath"
	"testing"
)

func TestColorAdjustmentsIsIdentity(t *testing.T) {
	identity := []ColorAdjustments{
		{},
		{Gamma: 1.0},
		{Gamma: 0},
		{Brightness: 0, Contrast: 0, Gamma: 1, Saturation: 0},
	}
	for i, adj := range identity {
		if !adj.IsIdentity() {
			t.Fatalf("case %d: %+v should be identity", i, adj)
		}
		if !adj.Valid() {
			t.Fatalf("case %d: identity config reported invalid", i)
		}
	}
	nonIdentity := []ColorAdjustments{
		{Brightness: 1},
		{Contrast: -1},
		{Gamma: 2.0},
		{Saturation: 1},
	}
	for i, adj := range nonIdentity {
		if adj.IsIdentity() {
			t.Fatalf("case %d: %+v should not be identity", i, adj)
		}
	}
}

func TestOpenResourceClosesFile(t *testing.T) {
	// Copy a fixture into a temp path and open it through the file API.
	dir := t.TempDir()
	dst := filepath.Join(dir, "img.png")
	data := openFixture(t, "odd_5x3.png")
	if err := os.WriteFile(dst, data, 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := OpenResource(dst)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Frames) != 1 || res.Width != 5 || res.Height != 3 {
		t.Fatalf("unexpected resource: %+v", res)
	}
	if _, err := OpenResource(filepath.Join(dir, "missing.png")); err == nil {
		t.Fatal("missing file must return an error")
	}
}
