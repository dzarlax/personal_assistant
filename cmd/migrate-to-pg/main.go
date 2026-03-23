// migrate-to-pg copies conversation data from SQLite to PostgreSQL.
//
// Usage:
//   DATABASE_URL="postgres://assistant_user:pass@host/aistack?search_path=assistant" \
//   go run ./cmd/migrate-to-pg --sqlite data/conversations.db
package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	_ "modernc.org/sqlite"
)

func main() {
	sqlitePath := flag.String("sqlite", "data/conversations.db", "path to SQLite database")
	dryRun := flag.Bool("dry-run", false, "count rows without migrating")
	flag.Parse()

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("DATABASE_URL environment variable is required")
	}

	// Open SQLite
	sdb, err := sql.Open("sqlite", *sqlitePath)
	if err != nil {
		log.Fatalf("open sqlite: %v", err)
	}
	defer sdb.Close()

	// Count rows
	var total int
	sdb.QueryRow("SELECT COUNT(*) FROM messages").Scan(&total)
	log.Printf("SQLite: %d messages to migrate", total)

	if *dryRun {
		return
	}

	// Connect to PostgreSQL
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("connect pg: %v", err)
	}
	defer pool.Close()

	// Verify target table exists (try a simple query — respects search_path)
	var tableExists bool
	err = pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM messages LIMIT 0)`).Scan(&tableExists)
	if err != nil {
		log.Fatalf("messages table does not exist in PostgreSQL (err: %v). Run init.sql first.", err)
	}

	// Check if target already has data
	var pgCount int
	pool.QueryRow(ctx, "SELECT COUNT(*) FROM messages").Scan(&pgCount)
	if pgCount > 0 {
		log.Printf("WARNING: PostgreSQL already has %d messages. Skipping migration to avoid duplicates.", pgCount)
		return
	}

	// Read all from SQLite
	rows, err := sdb.Query(`
		SELECT chat_id, role, content, parts, tool_calls, tool_call_id,
		       embedding, is_summary, is_compacted, is_reset, created_at
		FROM messages ORDER BY id ASC`)
	if err != nil {
		log.Fatalf("query sqlite: %v", err)
	}
	defer rows.Close()

	// Insert into PostgreSQL in batches
	batch := 0
	migrated := 0
	start := time.Now()

	tx, err := pool.Begin(ctx)
	if err != nil {
		log.Fatalf("begin tx: %v", err)
	}

	for rows.Next() {
		var chatID int64
		var role, content string
		var parts, toolCalls, toolCallID sql.NullString
		var embedding []byte
		var isSummary, isCompacted, isReset int
		var createdAt string

		if err := rows.Scan(&chatID, &role, &content, &parts, &toolCalls, &toolCallID,
			&embedding, &isSummary, &isCompacted, &isReset, &createdAt); err != nil {
			log.Printf("scan error (skipping): %v", err)
			continue
		}

		// Parse SQLite datetime (try multiple formats)
		var ts time.Time
		for _, layout := range []string{
			"2006-01-02 15:04:05",
			"2006-01-02T15:04:05Z",
			"2006-01-02 15:04:05-07:00",
			time.RFC3339,
		} {
			if t, err := time.Parse(layout, createdAt); err == nil {
				ts = t
				break
			}
		}
		if ts.IsZero() {
			log.Printf("WARNING: unparseable timestamp '%s' for row, using NOW()", createdAt)
			ts = time.Now()
		}

		// Convert nullable strings to *string for pgx
		var partsPtr, tcPtr, tcIDPtr *string
		if parts.Valid && parts.String != "" {
			partsPtr = &parts.String
		}
		if toolCalls.Valid && toolCalls.String != "" {
			tcPtr = &toolCalls.String
		}
		if toolCallID.Valid && toolCallID.String != "" {
			tcIDPtr = &toolCallID.String
		}

		// Convert int booleans to actual booleans
		var embBytes []byte
		if len(embedding) > 0 {
			embBytes = embedding
		}

		_, err := tx.Exec(ctx, `
			INSERT INTO messages (chat_id, role, content, parts, tool_calls, tool_call_id,
			                      embedding, is_summary, is_compacted, is_reset, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
			chatID, role, content, partsPtr, tcPtr, tcIDPtr,
			embBytes, isSummary != 0, isCompacted != 0, isReset != 0, ts)
		if err != nil {
			log.Printf("insert error (row %d): %v", migrated, err)
			continue
		}

		migrated++
		batch++
		if batch%500 == 0 {
			log.Printf("  migrated %d/%d...", migrated, total)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		log.Fatalf("commit: %v", err)
	}

	elapsed := time.Since(start)
	fmt.Printf("\nMigration complete: %d messages in %v\n", migrated, elapsed.Round(time.Millisecond))
}
