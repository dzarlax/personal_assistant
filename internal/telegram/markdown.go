package telegram

import (
	"fmt"
	"html"
	"regexp"
	"strings"
)

var (
	reFenceOpen   = regexp.MustCompile("^```(\\w*)")
	reInlineCode  = regexp.MustCompile("`([^`\n]+)`")
	reBoldStar    = regexp.MustCompile(`\*\*(.+?)\*\*`)
	reBoldUnder   = regexp.MustCompile(`__(.+?)__`)
	reItalicStar  = regexp.MustCompile(`\*([^*\n]+)\*`)
	reItalicUnder = regexp.MustCompile(`_([^_\n]+)_`)
	reStrike      = regexp.MustCompile(`~~(.+?)~~`)
	reLink        = regexp.MustCompile(`\[([^\]]+)\]\(([^)\s]+)\)`)
	reHeader      = regexp.MustCompile(`^#{1,6}\s+(.+)$`)
	reBulletItem  = regexp.MustCompile(`^(\s*)[-*]\s+(.+)$`)
	reNumItem     = regexp.MustCompile(`^(\s*\d+\.)\s+(.+)$`)
	rePlaceholder = regexp.MustCompile(`\x00(\d+)\x00`)
)

// markdownToTelegramHTML converts LLM Markdown output to Telegram-compatible HTML.
func markdownToTelegramHTML(src string) string {
	var sb strings.Builder
	lines := strings.Split(src, "\n")
	i := 0
	for i < len(lines) {
		line := lines[i]

		// Fenced code block
		if m := reFenceOpen.FindStringSubmatch(line); m != nil {
			lang := m[1]
			i++
			var codeLines []string
			for i < len(lines) && lines[i] != "```" {
				codeLines = append(codeLines, lines[i])
				i++
			}
			if i < len(lines) {
				i++ // skip closing ```
			}
			code := html.EscapeString(strings.Join(codeLines, "\n"))
			if lang != "" {
				sb.WriteString(fmt.Sprintf("<pre><code class=\"language-%s\">%s</code></pre>\n", html.EscapeString(lang), code))
			} else {
				sb.WriteString("<pre><code>" + code + "</code></pre>\n")
			}
			continue
		}

		sb.WriteString(processLine(line))
		sb.WriteByte('\n')
		i++
	}
	return strings.TrimSpace(sb.String())
}

func processLine(line string) string {
	if m := reHeader.FindStringSubmatch(line); m != nil {
		return "<b>" + processInline(m[1]) + "</b>"
	}
	if m := reBulletItem.FindStringSubmatch(line); m != nil {
		indent := strings.Repeat("  ", len(m[1])/2)
		return indent + "• " + processInline(m[2])
	}
	if m := reNumItem.FindStringSubmatch(line); m != nil {
		return html.EscapeString(m[1]) + " " + processInline(m[2])
	}
	return processInline(line)
}

func processInline(text string) string {
	// Extract inline code spans first (protect their content)
	var spans []string
	text = reInlineCode.ReplaceAllStringFunc(text, func(m string) string {
		inner := html.EscapeString(m[1 : len(m)-1])
		spans = append(spans, "<code>"+inner+"</code>")
		return fmt.Sprintf("\x00%d\x00", len(spans)-1)
	})

	// HTML-escape remaining text (*, _, ~, [, ], (, ) are not HTML special chars)
	text = escapeNonPlaceholders(text)

	// Apply formatting (order matters)
	text = reLink.ReplaceAllString(text, `<a href="$2">$1</a>`)
	text = reStrike.ReplaceAllStringFunc(text, func(m string) string {
		return "<s>" + m[2:len(m)-2] + "</s>"
	})
	text = reBoldStar.ReplaceAllStringFunc(text, func(m string) string {
		return "<b>" + m[2:len(m)-2] + "</b>"
	})
	text = reBoldUnder.ReplaceAllStringFunc(text, func(m string) string {
		return "<b>" + m[2:len(m)-2] + "</b>"
	})
	text = reItalicStar.ReplaceAllStringFunc(text, func(m string) string {
		return "<i>" + m[1:len(m)-1] + "</i>"
	})
	text = reItalicUnder.ReplaceAllStringFunc(text, func(m string) string {
		return "<i>" + m[1:len(m)-1] + "</i>"
	})

	// Restore code spans
	text = rePlaceholder.ReplaceAllStringFunc(text, func(m string) string {
		var idx int
		fmt.Sscanf(m[1:len(m)-1], "%d", &idx)
		if idx < len(spans) {
			return spans[idx]
		}
		return m
	})

	return text
}

func escapeNonPlaceholders(text string) string {
	locs := rePlaceholder.FindAllStringIndex(text, -1)
	if len(locs) == 0 {
		return html.EscapeString(text)
	}
	var sb strings.Builder
	last := 0
	for _, loc := range locs {
		sb.WriteString(html.EscapeString(text[last:loc[0]]))
		sb.WriteString(text[loc[0]:loc[1]])
		last = loc[1]
	}
	sb.WriteString(html.EscapeString(text[last:]))
	return sb.String()
}
