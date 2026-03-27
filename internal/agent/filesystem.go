package agent

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"telegram-agent/internal/llm"
)

const (
	fsListFilesTool      = "fs_list"
	fsReadFileTool       = "fs_read"
	fsWriteFileTool      = "fs_write"
	fsAppendFileTool     = "fs_append"
	fsDeleteFileTool     = "fs_delete"
	fsSearchFilesTool    = "fs_search"

	fsMaxReadSize   = 512 * 1024 // 512 KB
	fsMaxSearchHits = 200
)

// FilesystemConfig holds the root directory accessible to the agent.
type FilesystemConfig struct {
	Root string // absolute path; all operations are restricted to this directory
}

// safePath resolves a relative path inside the root directory.
// Returns an error if the resolved path escapes the root (path traversal protection).
func (c FilesystemConfig) safePath(relative string) (string, error) {
	// Clean and join
	joined := filepath.Join(c.Root, filepath.FromSlash(relative))
	abs, err := filepath.Abs(joined)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}
	// Ensure it's within root
	root, err := filepath.Abs(c.Root)
	if err != nil {
		return "", fmt.Errorf("invalid root: %w", err)
	}
	if abs != root && !strings.HasPrefix(abs, root+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes the allowed directory")
	}
	return abs, nil
}

// filesystemTools returns all tool definitions for LLM function calling.
func filesystemTools() []llm.Tool {
	return []llm.Tool{
		{
			Name:        fsListFilesTool,
			Description: "List files and directories in the context folder (or a subdirectory). Use this to explore what notes, references, and tasks are available.",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Relative path inside the context root. Empty string = root."
					}
				}
			}`),
		},
		{
			Name:        fsReadFileTool,
			Description: "Read the contents of a file from the context folder.",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Relative path to the file."
					}
				},
				"required": ["path"]
			}`),
		},
		{
			Name:        fsWriteFileTool,
			Description: "Write (create or overwrite) a file in the context folder.",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Relative path to the file."
					},
					"content": {
						"type": "string",
						"description": "Text content to write."
					}
				},
				"required": ["path", "content"]
			}`),
		},
		{
			Name:        fsAppendFileTool,
			Description: "Append text to a file in the context folder. Creates the file if it doesn't exist. Useful for adding notes or log entries without overwriting.",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Relative path to the file."
					},
					"content": {
						"type": "string",
						"description": "Text to append."
					}
				},
				"required": ["path", "content"]
			}`),
		},
		{
			Name:        fsDeleteFileTool,
			Description: "Delete a file from the context folder.",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "Relative path to the file to delete."
					}
				},
				"required": ["path"]
			}`),
		},
		{
			Name:        fsSearchFilesTool,
			Description: "Search for a text string inside files in the context folder. Returns matching lines with file path and line number.",
			InputSchema: json.RawMessage(`{
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Text to search for (case-insensitive)."
					},
					"path": {
						"type": "string",
						"description": "Subdirectory to search in. Empty = entire context root."
					}
				},
				"required": ["query"]
			}`),
		},
	}
}

// callFilesystem dispatches a filesystem tool call and returns the result string.
func callFilesystem(cfg FilesystemConfig, name, argsJSON string) (string, error) {
	switch name {
	case fsListFilesTool:
		return fsList(cfg, argsJSON)
	case fsReadFileTool:
		return fsRead(cfg, argsJSON)
	case fsWriteFileTool:
		return fsWrite(cfg, argsJSON)
	case fsAppendFileTool:
		return fsAppend(cfg, argsJSON)
	case fsDeleteFileTool:
		return fsDelete(cfg, argsJSON)
	case fsSearchFilesTool:
		return fsSearch(cfg, argsJSON)
	default:
		return "", fmt.Errorf("unknown filesystem tool: %s", name)
	}
}

func fsList(cfg FilesystemConfig, argsJSON string) (string, error) {
	var args struct {
		Path string `json:"path"`
	}
	_ = json.Unmarshal([]byte(argsJSON), &args)

	target, err := cfg.safePath(args.Path)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(target)
	if err != nil {
		return fmt.Sprintf("Path not found: %s", args.Path), nil
	}
	if !info.IsDir() {
		return fmt.Sprintf("Not a directory: %s", args.Path), nil
	}

	entries, err := os.ReadDir(target)
	if err != nil {
		return "", fmt.Errorf("read directory: %w", err)
	}
	if len(entries) == 0 {
		return "Directory is empty.", nil
	}

	var sb strings.Builder
	for _, e := range entries {
		rel, _ := filepath.Rel(cfg.Root, filepath.Join(target, e.Name()))
		if e.IsDir() {
			fmt.Fprintf(&sb, "[dir]  %s/\n", rel)
		} else {
			info, _ := e.Info()
			size := int64(0)
			if info != nil {
				size = info.Size()
			}
			fmt.Fprintf(&sb, "[file] %s  (%d bytes)\n", rel, size)
		}
	}
	return strings.TrimRight(sb.String(), "\n"), nil
}

func fsRead(cfg FilesystemConfig, argsJSON string) (string, error) {
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil || args.Path == "" {
		return "", fmt.Errorf("read_file: path is required")
	}

	target, err := cfg.safePath(args.Path)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(target)
	if os.IsNotExist(err) {
		return fmt.Sprintf("File not found: %s", args.Path), nil
	}
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return fmt.Sprintf("'%s' is a directory, not a file", args.Path), nil
	}
	if info.Size() > fsMaxReadSize {
		return fmt.Sprintf("File too large to read (%d bytes, max %d)", info.Size(), fsMaxReadSize), nil
	}

	data, err := os.ReadFile(target)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}
	return string(data), nil
}

func fsWrite(cfg FilesystemConfig, argsJSON string) (string, error) {
	var args struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil || args.Path == "" {
		return "", fmt.Errorf("write_file: path is required")
	}

	target, err := cfg.safePath(args.Path)
	if err != nil {
		return "", err
	}

	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return "", fmt.Errorf("create directories: %w", err)
	}
	if err := os.WriteFile(target, []byte(args.Content), 0o644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}
	return fmt.Sprintf("Written %d bytes to %s", len(args.Content), args.Path), nil
}

func fsAppend(cfg FilesystemConfig, argsJSON string) (string, error) {
	var args struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil || args.Path == "" {
		return "", fmt.Errorf("append_file: path is required")
	}

	target, err := cfg.safePath(args.Path)
	if err != nil {
		return "", err
	}

	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return "", fmt.Errorf("create directories: %w", err)
	}
	f, err := os.OpenFile(target, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return "", fmt.Errorf("open file: %w", err)
	}
	defer f.Close()
	if _, err := f.WriteString(args.Content); err != nil {
		return "", fmt.Errorf("append: %w", err)
	}
	return fmt.Sprintf("Appended %d bytes to %s", len(args.Content), args.Path), nil
}

func fsDelete(cfg FilesystemConfig, argsJSON string) (string, error) {
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil || args.Path == "" {
		return "", fmt.Errorf("delete_file: path is required")
	}

	target, err := cfg.safePath(args.Path)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(target)
	if os.IsNotExist(err) {
		return fmt.Sprintf("File not found: %s", args.Path), nil
	}
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return fmt.Sprintf("'%s' is a directory, not a file", args.Path), nil
	}
	if err := os.Remove(target); err != nil {
		return "", fmt.Errorf("delete: %w", err)
	}
	return fmt.Sprintf("Deleted: %s", args.Path), nil
}

func fsSearch(cfg FilesystemConfig, argsJSON string) (string, error) {
	var args struct {
		Query string `json:"query"`
		Path  string `json:"path"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil || args.Query == "" {
		return "", fmt.Errorf("search_files: query is required")
	}

	base, err := cfg.safePath(args.Path)
	if err != nil {
		return "", err
	}

	queryLower := strings.ToLower(args.Query)
	var hits []string

	err = filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil // skip unreadable files
		}
		lines := strings.Split(string(data), "\n")
		rel, _ := filepath.Rel(cfg.Root, path)
		for i, line := range lines {
			if strings.Contains(strings.ToLower(line), queryLower) {
				hits = append(hits, fmt.Sprintf("%s:%d: %s", rel, i+1, strings.TrimSpace(line)))
				if len(hits) >= fsMaxSearchHits {
					return filepath.SkipAll
				}
			}
		}
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("search: %w", err)
	}
	if len(hits) == 0 {
		return fmt.Sprintf("No matches for '%s'.", args.Query), nil
	}
	return strings.Join(hits, "\n"), nil
}
