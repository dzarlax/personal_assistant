#!/usr/bin/env bun
/**
 * pa-bridge — Custom Claude Code channel for Personal Assistant bot.
 *
 * MCP server (stdio) + HTTP server. PA bot sends messages via HTTP,
 * pa-bridge pushes them into Claude Code session. Claude replies via
 * the `reply` tool, which POSTs back to the PA bot's callback URL.
 *
 * Env:
 *   CALLBACK_URL  — PA bot callback (e.g. http://172.17.0.2:8444/claude-callback)
 *   PORT          — HTTP listen port (default 9800)
 *   OWNER_CHAT_ID — Telegram chat ID for permission relay
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const CALLBACK_URL = process.env.CALLBACK_URL || "";
const PORT = parseInt(process.env.PORT || "9800", 10);
const OWNER_CHAT_ID = process.env.OWNER_CHAT_ID || "0";

// --- Helpers ---

async function postCallback(payload: Record<string, string>) {
  if (!CALLBACK_URL) {
    console.error("[pa-bridge] No CALLBACK_URL, logging response:", JSON.stringify(payload));
    return;
  }
  try {
    await fetch(CALLBACK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.error("[pa-bridge] Callback failed:", err);
  }
}

// --- MCP Server ---

const mcp = new Server(
  { name: "pa-bridge", version: "0.1.0" },
  {
    capabilities: {
      experimental: {
        "claude/channel": {},
        "claude/channel/permission": {},
      },
      tools: {},
    },
    instructions: [
      'Messages arrive as <channel source="pa-bridge" chat_id="...">.',
      "Always reply using the reply tool, passing back the chat_id from the tag.",
      "Reply in the same language as the user's message.",
      "You are a personal assistant in Telegram — be concise and helpful.",
      "Do not mention that you are Claude Code or a CLI tool.",
    ].join(" "),
  }
);

// --- Reply tool ---

mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "reply",
      description: "Send a reply back to the user in Telegram",
      inputSchema: {
        type: "object" as const,
        properties: {
          chat_id: { type: "string", description: "The chat_id from the inbound channel tag" },
          text: { type: "string", description: "The message to send" },
        },
        required: ["chat_id", "text"],
      },
    },
  ],
}));

mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
  if (req.params.name === "reply") {
    const { chat_id, text } = req.params.arguments as { chat_id: string; text: string };
    await postCallback({ type: "reply", chat_id, text });
    return { content: [{ type: "text" as const, text: "sent" }] };
  }
  throw new Error(`Unknown tool: ${req.params.name}`);
});

// --- Permission relay ---

const PermissionRequestSchema = z.object({
  method: z.literal("notifications/claude/channel/permission_request"),
  params: z.object({
    request_id: z.string(),
    tool_name: z.string(),
    description: z.string(),
    input_preview: z.string(),
  }),
});

mcp.setNotificationHandler(PermissionRequestSchema, async ({ params }) => {
  await postCallback({
    type: "permission",
    chat_id: OWNER_CHAT_ID,
    request_id: params.request_id,
    text: `${params.tool_name}: ${params.description}`,
  });
});

// --- Connect stdio ---

await mcp.connect(new StdioServerTransport());
console.error(`[pa-bridge] MCP connected, HTTP on :${PORT}`);

// --- HTTP server ---

Bun.serve({
  port: PORT,
  hostname: "127.0.0.1",
  async fetch(req) {
    const url = new URL(req.url);

    // PA bot sends user message
    if (req.method === "POST" && url.pathname === "/message") {
      try {
        const { chat_id, text } = (await req.json()) as { chat_id: string; text: string };
        if (!chat_id || !text) {
          return Response.json({ error: "chat_id and text required" }, { status: 400 });
        }
        await mcp.notification({
          method: "notifications/claude/channel",
          params: { content: text, meta: { chat_id } },
        });
        return Response.json({ ok: true });
      } catch (err) {
        console.error("[pa-bridge] /message error:", err);
        return Response.json({ error: "internal error" }, { status: 500 });
      }
    }

    // PA bot sends permission verdict
    if (req.method === "POST" && url.pathname === "/permission") {
      try {
        const { request_id, behavior } = (await req.json()) as {
          request_id: string;
          behavior: string;
        };
        if (!request_id || !behavior) {
          return Response.json({ error: "request_id and behavior required" }, { status: 400 });
        }
        await mcp.notification({
          method: "notifications/claude/channel/permission",
          params: { request_id, behavior },
        });
        return Response.json({ ok: true });
      } catch (err) {
        console.error("[pa-bridge] /permission error:", err);
        return Response.json({ error: "internal error" }, { status: 500 });
      }
    }

    // Health check
    if (req.method === "GET" && url.pathname === "/health") {
      return Response.json({ status: "ok" });
    }

    return new Response("not found", { status: 404 });
  },
});
