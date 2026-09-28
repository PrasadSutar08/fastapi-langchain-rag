import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { createRoot } from "react-dom/client";
import * as api from "./api";
import "./styles.css";

type ChatMessage = {
  id?: number;
  role: "user" | "assistant";
  content: string;
  sources?: api.Source[];
};

const DEFAULT_TITLE = "New Conversation";

/** Deterministic fallback title used only if the /auto-title network
 *  call itself fails (LLM-side echo/failure is already handled
 *  server-side) -- strips common leading question phrasing rather
 *  than blindly truncating, e.g. "what is this document about?"
 *  becomes "this document about" instead of a verbatim clip. */
const LEADING_PATTERNS = [
  /^(please\s+)?(tell me|explain|describe|summarize)\s+/i,
  /^(how)\s+(many|much|long|often|far)\s+/i,
  /^(what|how|why|when|where|who|which)\s+(is|are|was|were|do|does|did|can|could|would|should)\s+/i,
  /^(is|are|do|does|did|can|could|would|should)\s+/i,
];

function summarize(text: string, maxWords = 6): string {
  let clean = text.trim().replace(/\s+/g, " ");

  for (const pattern of LEADING_PATTERNS) {
    const match = clean.match(pattern);
    if (match) {
      clean = clean.slice(match[0].length).trim();
      break;
    }
  }

  clean = clean.replace(/^[\s?!."']+|[\s?!."']+$/g, "");
  if (!clean) clean = text.trim().replace(/^[\s?!."']+|[\s?!."']+$/g, "");

  const words = clean.split(/\s+/).slice(0, maxWords);
  const title = words.join(" ");
  if (!title) return "New Conversation";
  return title[0].toUpperCase() + title.slice(1);
}

/** Multiple chunks from the same page produce duplicate-looking
 *  citation chips (same filename + page shown twice); collapse them. */
function dedupeSources(sources: api.Source[]): api.Source[] {
  const seen = new Set<string>();
  const result: api.Source[] = [];
  for (const s of sources) {
    const key = `${s.filename ?? ""}::${s.page ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(s);
  }
  return result;
}

function Login({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await api.login(email, password);
      onLogin();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth">
      <form className="card auth-card" onSubmit={submit}>
        <div className="brand">
          Doc<span>RAG</span>
        </div>
        <h1>Sign in</h1>
        <p className="muted">Secure document intelligence workspace.</p>
        {error && <div className="error">{error}</div>}
        <label>
          Email
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            type="email"
            required
          />
        </label>
        <label>
          Password
          <input
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            type="password"
            required
          />
        </label>
        <button disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
      </form>
    </main>
  );
}

/** Assistant "thinking" indicator with a running elapsed-time readout,
 *  since responses can legitimately take 10-90+ seconds against a
 *  local LLM -- a bare spinner with no feedback reads as "stuck". */
function TypingIndicator() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <article className="message assistant">
      <div className="bubble typing">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
        <span className="typing-elapsed">Generating… {seconds}s</span>
      </div>
    </article>
  );
}

function App() {
  const [authenticated, setAuthenticated] = useState(Boolean(api.getToken()));
  const [conversations, setConversations] = useState<api.Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<api.DocumentItem[]>([]);
  const [input, setInput] = useState("");
  // Per-conversation "generating" state (a Set of conversation ids
  // currently waiting on an LLM response), rather than one global
  // flag -- so sending a message in one conversation doesn't lock the
  // composer in every other conversation while Ollama is thinking.
  const [pendingIds, setPendingIds] = useState<Set<number>>(new Set());
  const busy = activeId !== null && pendingIds.has(activeId);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [results, setResults] = useState<api.SearchResult[]>([]);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editingText, setEditingText] = useState("");
  const active = useMemo(
    () => conversations.find((c) => c.id === activeId),
    [conversations, activeId]
  );

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const activeIdRef = useRef<number | null>(null);
  activeIdRef.current = activeId;

  async function load() {
    try {
      const [c, d] = await Promise.all([
        api.listConversations(),
        api.listDocuments(),
      ]);
      setConversations(c.data);
      setDocuments(d.documents);
      if (!activeId && c.data[0]) setActiveId(c.data[0].id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  useEffect(() => {
    const h = () => setAuthenticated(false);
    window.addEventListener("auth-expired", h);
    if (authenticated) void load();
    return () => window.removeEventListener("auth-expired", h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authenticated]);

  // Loads the message history for whichever conversation is active.
  // Guarded against races: if the user switches conversations again
  // before this fetch resolves, the stale response is discarded
  // instead of overwriting the (now different) active conversation's
  // messages -- this was the cause of history appearing to "wipe"
  // when switching quickly between conversations.
  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    const requestedId = activeId;
    api
      .listMessages(requestedId)
      .then((r) => {
        if (cancelled || activeIdRef.current !== requestedId) return;
        setMessages(
          r.data.map((m) => ({
            id: m.id,
            role: m.role === "user" ? "user" : "assistant",
            content: m.content,
          }))
        );
      })
      .catch((e) => {
        if (!cancelled) setError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  // Auto-scroll to the newest message / typing indicator. Uses "auto"
  // rather than "smooth": with the 1s ticking typing-indicator update
  // and streaming-length answers, a smooth scroll gets re-triggered
  // and interrupted repeatedly, which is what produced the sluggish
  // feeling -- an instant jump reads as snappier here.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [messages, busy]);

  async function newChat() {
    try {
      const r = await api.createConversation();
      setConversations((c) => [r.data, ...c]);
      setActiveId(r.data.id);
      setMessages([]);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function removeChat(id: number) {
    try {
      await api.deleteConversation(id);
      const next = conversations.filter((c) => c.id !== id);
      setConversations(next);
      setActiveId(next[0]?.id ?? null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  /** Rename a still-default-titled conversation using an LLM-generated
   *  short (2-6 word) summary of the message, the way Claude.ai titles
   *  new chats -- falls back to a plain truncation if the LLM call
   *  fails, so a title always gets set either way. */
  async function maybeAutoTitle(conversationId: number, firstMessage: string) {
    const conversation = conversations.find((c) => c.id === conversationId);
    if (!conversation || conversation.title !== DEFAULT_TITLE) return;

    try {
      const r = await api.autoTitleConversation(conversationId, firstMessage);
      setConversations((cs) =>
        cs.map((c) => (c.id === conversationId ? r.data : c))
      );
    } catch {
      const title = summarize(firstMessage);
      setConversations((cs) =>
        cs.map((c) => (c.id === conversationId ? { ...c, title } : c))
      );
    }
  }

  /** Sends `question` to a conversation and appends the reply.
   *  `userMessageIndex`, if given, is the index of the just-appended
   *  optimistic user bubble in `messages` -- once the real message id
   *  comes back from the server, it's patched onto that bubble so
   *  edit/regenerate (which need a real id) work on freshly-sent
   *  messages too, not just ones reloaded from history. */
  async function runChat(
    conversationId: number,
    question: string,
    userMessageIndex?: number
  ) {
    setError("");
    setPendingIds((s) => new Set(s).add(conversationId));
    try {
      const r = await api.chat(conversationId, question);
      if (activeIdRef.current === conversationId) {
        setMessages((m) => {
          const next = [...m];
          if (userMessageIndex !== undefined && next[userMessageIndex]) {
            next[userMessageIndex] = {
              ...next[userMessageIndex],
              id: r.data.user_message_id,
            };
          }
          next.push({
            id: r.data.assistant_message_id,
            role: "assistant",
            content: r.data.answer,
            sources: r.data.sources,
          });
          return next;
        });
      }
      void maybeAutoTitle(conversationId, question);
    } catch (e) {
      if (activeIdRef.current === conversationId) setError((e as Error).message);
    } finally {
      setPendingIds((s) => {
        const next = new Set(s);
        next.delete(conversationId);
        return next;
      });
    }
  }

  async function send(e: FormEvent) {
    e.preventDefault();
    if (!activeId || !input.trim() || pendingIds.has(activeId)) return;
    const conversationId = activeId;
    const q = input.trim();
    const userMessageIndex = messages.length;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    void runChat(conversationId, q, userMessageIndex);
  }

  /** Re-ask the question that produced the assistant message at
   *  `index`, discarding that answer (and anything after it) first. */
  function regenerate(index: number) {
    if (!activeId || pendingIds.has(activeId)) return;
    const priorUserMessage = [...messages.slice(0, index)]
      .reverse()
      .find((m) => m.role === "user");
    if (!priorUserMessage) return;
    setMessages((m) => m.slice(0, index));
    void runChat(activeId, priorUserMessage.content);
  }

  /** Save an edited user message: deletes it (and everything after)
   *  server-side, then resends the edited text as a fresh turn --
   *  same UX as editing a message in Claude.ai. */
  async function saveEdit(index: number) {
    if (!activeId) return;
    const conversationId = activeId;
    const original = messages[index];
    const newText = editingText.trim();
    if (!newText) return;

    setEditingIndex(null);

    if (original.id != null) {
      try {
        await api.deleteMessagesFrom(conversationId, original.id);
      } catch (e) {
        setError((e as Error).message);
        return;
      }
    }

    const userMessageIndex = index;
    setMessages((m) => [...m.slice(0, index), { role: "user", content: newText }]);
    void runChat(conversationId, newText, userMessageIndex);
  }

  async function copyMessage(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard API unavailable (e.g. insecure context) -- silently
      // ignore rather than surface a confusing error for a nice-to-have.
    }
  }

    /** Download the active conversation as a plain-text transcript,
   *  including cited source pages under each answer. */
  function exportChat() {
    if (!active || !messages.length) return;

    const lines: string[] = [];
    lines.push(active.title);
    lines.push(`Exported ${new Date().toLocaleString()}`);
    lines.push("=".repeat(40));
    lines.push("");

    for (const m of messages) {
      lines.push(m.role === "user" ? "You:" : "Assistant:");
      lines.push(m.content);
      if (m.sources?.length) {
        const cited = dedupeSources(m.sources)
          .map((s) => `${s.filename || "Document"}${s.page ? ` (p. ${s.page})` : ""}`)
          .join(", ");
        lines.push(`Sources: ${cited}`);
      }
      lines.push("");
    }

    const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeName = active.title.replace(/[^\w\- ]+/g, "").trim() || "conversation";
    a.href = url;
    a.download = `${safeName}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function upload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      await api.uploadDocument(file);
      setDocuments((await api.listDocuments()).documents);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function removeDoc(id: number) {
    try {
      await api.deleteDocument(id);
      setDocuments((d) => d.filter((x) => x.id !== id));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function doSearch(e: FormEvent) {
    e.preventDefault();
    if (!search.trim()) return;
    try {
      setResults((await api.searchDocuments(search.trim())).results);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (!authenticated) return <Login onLogin={() => setAuthenticated(true)} />;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-scroll">
          <div className="brand">
            Doc<span>RAG</span>
          </div>
          <button className="new" onClick={newChat}>
            + New conversation
          </button>
          <div className="side-title">Conversations</div>
          <div className="conversations">
            {conversations.map((c) => (
              <div
                className={`conversation ${c.id === activeId ? "active" : ""}`}
                key={c.id}
                onClick={() => setActiveId(c.id)}
              >
                <span>{c.title}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    void removeChat(c.id);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
        <div className="side-bottom">
          <label className={`upload ${uploading ? "busy" : ""}`}>
            {uploading ? (
              <>
                <span className="spinner" /> Uploading…
              </>
            ) : (
              "Upload PDF"
            )}
            <input
              hidden
              type="file"
              accept="application/pdf,.pdf"
              onChange={upload}
              disabled={uploading}
            />
          </label>
          <button
            className="ghost"
            onClick={() => {
              api.clearToken();
              setAuthenticated(false);
            }}
          >
            Sign out
          </button>
        </div>
      </aside>
      <main className="main">
        <header>
          <div>
            <h1>{active?.title || "Document Intelligence"}</h1>
            <p className="muted">Ask questions grounded in your uploaded PDFs.</p>
          </div>
          <button
            type="button"
            className="pill export-btn"
            onClick={exportChat}
            disabled={!messages.length}
            title={messages.length ? "Download this conversation" : "No messages to export yet"}
          >
            Export chat
          </button>
        </header>
        {error && (
          <div className="error banner">
            {error}
            <button onClick={() => setError("")}>×</button>
          </div>
        )}
        <div className="grid">
          <section className="chat card">
            <div className="messages">
              {!messages.length && !busy && (
                <div className="empty">
                  <h2>Start a grounded conversation</h2>
                  <p>
                    Upload a PDF, then ask a precise question. Answers include
                    source pages when available.
                  </p>
                </div>
              )}
              {messages.map((m, i) => (
                <article className={`message ${m.role}`} key={i}>
                  {m.role === "user" && editingIndex === i ? (
                    <div className="edit-box">
                      <textarea
                        value={editingText}
                        onChange={(e) => setEditingText(e.target.value)}
                        rows={Math.min(6, Math.max(2, Math.ceil(editingText.length / 40)))}
                        autoFocus
                      />
                      <div className="edit-actions">
                        <button type="button" onClick={() => setEditingIndex(null)}>
                          Cancel
                        </button>
                        <button
                          type="button"
                          className="primary"
                          onClick={() => void saveEdit(i)}
                          disabled={!editingText.trim()}
                        >
                          Save &amp; submit
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="bubble">{m.content}</div>
                  )}
                  {m.sources?.length ? (
                    <div className="sources">
                      {dedupeSources(m.sources).map((s, j) => (
                        <span key={j}>
                          {s.filename || "Document"}
                          {s.page ? ` · p. ${s.page}` : ""}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {editingIndex !== i && (
                    <div className="message-actions">
                      <button
                        type="button"
                        className="icon-action"
                        onClick={() => void copyMessage(m.content)}
                      >
                        Copy
                      </button>
                      {m.role === "user" && (
                        <button
                          type="button"
                          className="icon-action"
                          disabled={busy}
                          onClick={() => {
                            setEditingIndex(i);
                            setEditingText(m.content);
                          }}
                        >
                          Edit
                        </button>
                      )}
                      {m.role === "assistant" && (
                        <button
                          type="button"
                          className="icon-action"
                          onClick={() => regenerate(i)}
                          disabled={busy}
                        >
                          Regenerate
                        </button>
                      )}
                    </div>
                  )}
                </article>
              ))}
              {busy && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
            <form className="composer" onSubmit={send}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={!activeId || busy}
                placeholder={
                  activeId ? "Ask about your documents…" : "Create a conversation first"
                }
              />
              <button disabled={!activeId || busy || !input.trim()}>
                {busy ? "Generating…" : "Send"}
              </button>
            </form>
          </section>
          <aside className="right">
            <section className="card panel">
              <div className="panel-head">
                <h2>Documents</h2>
                <span>{documents.length}</span>
              </div>
              {documents.length ? (
                documents.map((d) => (
                  <div className="doc" key={d.id}>
                    <div>
                      <strong title={d.filename}>{d.filename}</strong>
                      <small>
                        {d.status} · {d.chunk_count} chunks
                      </small>
                    </div>
                    <button onClick={() => void removeDoc(d.id)}>Delete</button>
                  </div>
                ))
              ) : (
                <p className="muted">No documents uploaded.</p>
              )}
            </section>
            <section className="card panel">
              <h2>Semantic search</h2>
              <form className="search" onSubmit={doSearch}>
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search your PDFs"
                />
                <button>Search</button>
              </form>
              {results.map((r, i) => (
                <div className="result" key={i}>
                  <strong>
                    {r.filename || "Document"}
                    {r.page ? ` · p. ${r.page}` : ""}
                  </strong>
                  <p>{r.content}</p>
                </div>
              ))}
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
