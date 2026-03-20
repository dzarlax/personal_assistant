package telegram

import (
	"testing"
)

func TestMarkdown_PlainText(t *testing.T) {
	assertMD(t, "hello world", "hello world")
}

func TestMarkdown_Empty(t *testing.T) {
	assertMD(t, "", "")
}

// --- Headers ---

func TestMarkdown_H1(t *testing.T) {
	assertMD(t, "# Title", "<b>Title</b>")
}

func TestMarkdown_H3(t *testing.T) {
	assertMD(t, "### Section", "<b>Section</b>")
}

// --- Bold ---

func TestMarkdown_BoldStars(t *testing.T) {
	assertMD(t, "**bold**", "<b>bold</b>")
}

func TestMarkdown_BoldUnderscores(t *testing.T) {
	assertMD(t, "__bold__", "<b>bold</b>")
}

// --- Italic ---

func TestMarkdown_ItalicStar(t *testing.T) {
	assertMD(t, "*italic*", "<i>italic</i>")
}

func TestMarkdown_ItalicUnderscore(t *testing.T) {
	assertMD(t, "_italic_", "<i>italic</i>")
}

// --- Strikethrough ---

func TestMarkdown_Strikethrough(t *testing.T) {
	assertMD(t, "~~strike~~", "<s>strike</s>")
}

// --- Inline code ---

func TestMarkdown_InlineCode(t *testing.T) {
	assertMD(t, "`code`", "<code>code</code>")
}

func TestMarkdown_InlineCodeProtectsContent(t *testing.T) {
	// Markdown inside inline code must NOT be formatted
	assertMD(t, "`**not bold**`", "<code>**not bold**</code>")
}

func TestMarkdown_InlineCodeEscapesHTML(t *testing.T) {
	assertMD(t, "`<div>`", "<code>&lt;div&gt;</code>")
}

// --- Links ---

func TestMarkdown_Link(t *testing.T) {
	assertMD(t, "[OpenAI](https://openai.com)", `<a href="https://openai.com">OpenAI</a>`)
}

// --- Fenced code blocks ---

func TestMarkdown_FencedCodeBlock(t *testing.T) {
	input := "```go\nfmt.Println(\"hi\")\n```"
	want := "<pre><code class=\"language-go\">fmt.Println(&#34;hi&#34;)</code></pre>"
	assertMD(t, input, want)
}

func TestMarkdown_FencedCodeBlockNoLang(t *testing.T) {
	input := "```\nsome code\n```"
	want := "<pre><code>some code</code></pre>"
	assertMD(t, input, want)
}

func TestMarkdown_FencedCodeBlockEscapesHTML(t *testing.T) {
	input := "```\n<script>alert(1)</script>\n```"
	want := "<pre><code>&lt;script&gt;alert(1)&lt;/script&gt;</code></pre>"
	assertMD(t, input, want)
}

// --- Lists ---

func TestMarkdown_BulletDash(t *testing.T) {
	assertMD(t, "- item", "• item")
}

func TestMarkdown_BulletStar(t *testing.T) {
	assertMD(t, "* item", "• item")
}

func TestMarkdown_NumberedList(t *testing.T) {
	assertMD(t, "1. first", "1. first")
}

func TestMarkdown_BulletListMultiline(t *testing.T) {
	input := "- one\n- two\n- three"
	want := "• one\n• two\n• three"
	assertMD(t, input, want)
}

// --- Checkboxes ---

func TestMarkdown_CheckboxUnchecked(t *testing.T) {
	assertMD(t, "- [ ] task", "☐ task")
}

func TestMarkdown_CheckboxChecked(t *testing.T) {
	assertMD(t, "- [x] done", "☑ done")
}

func TestMarkdown_CheckboxCheckedUpper(t *testing.T) {
	assertMD(t, "* [X] done", "☑ done")
}

func TestMarkdown_CheckboxWithBold(t *testing.T) {
	assertMD(t, "*   [ ] **Паспорт** (загран)", "☐ <b>Паспорт</b> (загран)")
}

// --- HTML escaping ---

func TestMarkdown_HTMLInText(t *testing.T) {
	assertMD(t, "use <b> tag", "use &lt;b&gt; tag")
}

func TestMarkdown_AmpersandEscaped(t *testing.T) {
	assertMD(t, "a & b", "a &amp; b")
}

// --- Mixed content ---

func TestMarkdown_BoldInHeader(t *testing.T) {
	assertMD(t, "## **Section**", "<b><b>Section</b></b>")
}

func TestMarkdown_InlineCodeInBoldNotFormatted(t *testing.T) {
	// Code span is extracted before bold, so it's protected
	got := markdownToTelegramHTML("**use `code` here**")
	if got == "" {
		t.Fatal("got empty result")
	}
	// Should contain <code>code</code> and <b>...</b>
	if !contains(got, "<code>code</code>") {
		t.Errorf("expected <code>code</code> in %q", got)
	}
}

func TestMarkdown_MultilineDocument(t *testing.T) {
	input := "# Title\n\nSome **bold** text.\n\n- item 1\n- item 2"
	got := markdownToTelegramHTML(input)
	if !contains(got, "<b>Title</b>") {
		t.Errorf("missing header in %q", got)
	}
	if !contains(got, "<b>bold</b>") {
		t.Errorf("missing bold in %q", got)
	}
	if !contains(got, "• item 1") {
		t.Errorf("missing bullet in %q", got)
	}
}

// --- splitMessage ---

func TestSplitMessage_Short(t *testing.T) {
	parts := splitMessage("hello", 100)
	if len(parts) != 1 || parts[0] != "hello" {
		t.Errorf("unexpected: %v", parts)
	}
}

func TestSplitMessage_ParagraphBoundary(t *testing.T) {
	text := "first paragraph\n\nsecond paragraph\n\nthird paragraph"
	parts := splitMessage(text, 30)
	if len(parts) < 2 {
		t.Errorf("expected multiple parts, got %d: %v", len(parts), parts)
	}
	// Verify no part exceeds maxLen
	for i, p := range parts {
		if len(p) > 30 {
			t.Errorf("part %d exceeds maxLen: %d chars", i, len(p))
		}
	}
}

func TestSplitMessage_LineBoundary(t *testing.T) {
	text := "line one\nline two\nline three\nline four"
	parts := splitMessage(text, 20)
	if len(parts) < 2 {
		t.Errorf("expected multiple parts, got %d: %v", len(parts), parts)
	}
	for i, p := range parts {
		if len(p) > 20 {
			t.Errorf("part %d exceeds maxLen: %d chars", i, len(p))
		}
	}
}

// --- helpers ---

func assertMD(t *testing.T, input, want string) {
	t.Helper()
	got := markdownToTelegramHTML(input)
	if got != want {
		t.Errorf("\ninput: %q\n  got: %q\n want: %q", input, got, want)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(sub) == 0 ||
		func() bool {
			for i := 0; i <= len(s)-len(sub); i++ {
				if s[i:i+len(sub)] == sub {
					return true
				}
			}
			return false
		}())
}
