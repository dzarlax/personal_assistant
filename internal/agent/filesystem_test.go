package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func tmpRoot(t *testing.T) (FilesystemConfig, string) {
	t.Helper()
	dir := t.TempDir()
	return FilesystemConfig{Root: dir}, dir
}

// --- safePath tests ---

func TestSafePath_Normal(t *testing.T) {
	cfg, root := tmpRoot(t)
	got, err := cfg.safePath("notes/file.md")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(root, "notes", "file.md")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestSafePath_Root(t *testing.T) {
	cfg, root := tmpRoot(t)
	got, err := cfg.safePath("")
	if err != nil {
		t.Fatal(err)
	}
	if got != root {
		t.Errorf("got %q, want %q", got, root)
	}
}

func TestSafePath_TraversalBlocked(t *testing.T) {
	cfg, _ := tmpRoot(t)
	cases := []string{
		"../etc/passwd",
		"notes/../../etc/passwd",
		"../",
		"..",
	}
	for _, c := range cases {
		_, err := cfg.safePath(c)
		if err == nil {
			t.Errorf("safePath(%q) should have returned error", c)
		}
	}
}

func TestSafePath_DotSlash(t *testing.T) {
	cfg, root := tmpRoot(t)
	got, err := cfg.safePath("./notes/../notes/file.md")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(root, "notes", "file.md")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// --- Tool integration tests ---

func TestListFiles_Empty(t *testing.T) {
	cfg, _ := tmpRoot(t)
	result, err := fsList(cfg, `{}`)
	if err != nil {
		t.Fatal(err)
	}
	if result != "Directory is empty." {
		t.Errorf("unexpected: %q", result)
	}
}

func TestListFiles_WithEntries(t *testing.T) {
	cfg, root := tmpRoot(t)
	os.MkdirAll(filepath.Join(root, "notes"), 0o755)
	os.WriteFile(filepath.Join(root, "readme.md"), []byte("hello"), 0o644)

	result, err := fsList(cfg, `{}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) == 0 {
		t.Fatal("expected non-empty result")
	}
}

func TestWriteAndReadFile(t *testing.T) {
	cfg, _ := tmpRoot(t)

	args, _ := json.Marshal(map[string]string{
		"path":    "notes/test.md",
		"content": "hello world",
	})
	result, err := fsWrite(cfg, string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result == "" {
		t.Fatal("expected non-empty result")
	}

	readArgs, _ := json.Marshal(map[string]string{"path": "notes/test.md"})
	content, err := fsRead(cfg, string(readArgs))
	if err != nil {
		t.Fatal(err)
	}
	if content != "hello world" {
		t.Errorf("got %q, want %q", content, "hello world")
	}
}

func TestAppendFile(t *testing.T) {
	cfg, _ := tmpRoot(t)

	args1, _ := json.Marshal(map[string]string{"path": "log.txt", "content": "line1\n"})
	args2, _ := json.Marshal(map[string]string{"path": "log.txt", "content": "line2\n"})

	if _, err := fsAppend(cfg, string(args1)); err != nil {
		t.Fatal(err)
	}
	if _, err := fsAppend(cfg, string(args2)); err != nil {
		t.Fatal(err)
	}

	readArgs, _ := json.Marshal(map[string]string{"path": "log.txt"})
	content, err := fsRead(cfg, string(readArgs))
	if err != nil {
		t.Fatal(err)
	}
	if content != "line1\nline2\n" {
		t.Errorf("got %q", content)
	}
}

func TestDeleteFile(t *testing.T) {
	cfg, root := tmpRoot(t)
	os.WriteFile(filepath.Join(root, "tmp.txt"), []byte("x"), 0o644)

	args, _ := json.Marshal(map[string]string{"path": "tmp.txt"})
	result, err := fsDelete(cfg, string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result != "Deleted: tmp.txt" {
		t.Errorf("unexpected: %q", result)
	}
	if _, err := os.Stat(filepath.Join(root, "tmp.txt")); !os.IsNotExist(err) {
		t.Error("file should be deleted")
	}
}

func TestDeleteFile_NotFound(t *testing.T) {
	cfg, _ := tmpRoot(t)
	args, _ := json.Marshal(map[string]string{"path": "nope.txt"})
	result, err := fsDelete(cfg, string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result != "File not found: nope.txt" {
		t.Errorf("unexpected: %q", result)
	}
}

func TestSearchFiles(t *testing.T) {
	cfg, root := tmpRoot(t)
	os.MkdirAll(filepath.Join(root, "notes"), 0o755)
	os.WriteFile(filepath.Join(root, "notes", "a.md"), []byte("line one\nfind me here\nline three"), 0o644)
	os.WriteFile(filepath.Join(root, "notes", "b.md"), []byte("nothing relevant"), 0o644)

	args, _ := json.Marshal(map[string]string{"query": "find me"})
	result, err := fsSearch(cfg, string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result == "" || result == "No matches for 'find me'." {
		t.Errorf("expected match, got: %q", result)
	}
}

func TestSearchFiles_CaseInsensitive(t *testing.T) {
	cfg, root := tmpRoot(t)
	os.WriteFile(filepath.Join(root, "test.md"), []byte("Hello World"), 0o644)

	args, _ := json.Marshal(map[string]string{"query": "hello world"})
	result, err := fsSearch(cfg, string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result == "No matches for 'hello world'." {
		t.Error("case-insensitive search should have matched")
	}
}

func TestWriteFile_TraversalBlocked(t *testing.T) {
	cfg, _ := tmpRoot(t)
	args, _ := json.Marshal(map[string]string{"path": "../escape.txt", "content": "pwned"})
	_, err := fsWrite(cfg, string(args))
	if err == nil {
		t.Error("write outside root should fail")
	}
}

func TestReadFile_TooLarge(t *testing.T) {
	cfg, root := tmpRoot(t)
	big := make([]byte, fsMaxReadSize+1)
	os.WriteFile(filepath.Join(root, "big.bin"), big, 0o644)

	args, _ := json.Marshal(map[string]string{"path": "big.bin"})
	result, err := fsRead(cfg, string(args))
	if err != nil {
		t.Fatal(err)
	}
	if result == "" || result == string(big) {
		t.Error("should return size error, not contents")
	}
}
