import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Download,
  Check,
  FileText,
  Lightbulb,
  LoaderCircle,
  ListChecks,
  Menu,
  MessageSquare,
  Palette,
  Paperclip,
  Plus,
  Sparkles,
  X,
} from "lucide-react";

const suggestions = [
  { title: "Summarize a document", prompt: "Summarize the uploaded document.", icon: FileText },
  { title: "Find key insights", prompt: "What are the key takeaways?", icon: ListChecks },
  { title: "Explain a concept", prompt: "Explain the main ideas in simple terms.", icon: Lightbulb },
];

const themes = [
  { id: "ember", name: "Ember Night", description: "Warm peach on black." },
  { id: "violet", name: "Violet", description: "Deep indigo with lavender light." },
  { id: "crimson", name: "Crimson & Gold", description: "A red-black glow, gold buttons. The default." },
  { id: "charcoal", name: "Charcoal & Gold", description: "Plain black with a gold accent." },
];

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function normalizeAssistantMarkdown(content) {
  const markdown = typeof content === "string" ? content : String(content ?? "");
  return markdown.replace(/<br\s*\/?>/gi, (tag, offset, source) => {
    const lineStart = source.lastIndexOf("\n", offset) + 1;
    const nextLineBreak = source.indexOf("\n", offset);
    const lineEnd = nextLineBreak < 0 ? source.length : nextLineBreak;
    const line = source.slice(lineStart, lineEnd);

    // Keep generated table rows intact; use a readable separator within the cell.
    return /^\s*\|.*\|\s*$/.test(line) ? " · " : "\n\n";
  });
}

function App() {
  const [threads, setThreads] = useState([]);
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [file, setFile] = useState(null);
  const [uploadedDocument, setUploadedDocument] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [theme, setTheme] = useState(() => localStorage.getItem("nova-theme") || "crimson");
  const [themeOpen, setThemeOpen] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const themeMenuRef = useRef(null);

  useEffect(() => {
    api("/threads")
      .then(setThreads)
      .catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("nova-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!themeOpen) return undefined;
    function closeOnOutsideClick(event) {
      if (!themeMenuRef.current?.contains(event.target)) setThemeOpen(false);
    }
    function closeOnEscape(event) {
      if (event.key === "Escape") setThemeOpen(false);
    }
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [themeOpen]);

  async function openThread(id) {
    setThreadId(id);
    setError("");
    setFile(null);
    setSidebarOpen(false);
    try {
      const data = await api(`/threads/${id}`);
      setMessages(data.messages);
      setUploadedDocument(data.document?.filename ? data.document : null);
    } catch (reason) {
      setMessages([]);
      setUploadedDocument(null);
      setError(reason.message);
    }
  }

  async function newThread() {
    try {
      const thread = await api("/threads", { method: "POST" });
      setThreadId(thread.id);
      setMessages([]);
      setUploadedDocument(null);
      setFile(null);
      setDraft("");
      setError("");
      setSidebarOpen(false);
    } catch (reason) {
      setError(reason.message);
    }
  }

  async function sendMessage(text = draft) {
    const prompt = text.trim();
    if ((!prompt && !file) || busy || uploading) return;
    setError("");
    setDraft("");
    setBusy(true);

    let nextMessages = messages;
    try {
      if (file) {
        setUploading(true);
        const form = new FormData();
        form.append("file", file);
        const indexed = await api(`/threads/${threadId}/pdf`, {
          method: "POST",
          body: form,
        });
        setUploadedDocument(indexed);
        setFile(null);
        setFileInputKey((value) => value + 1);
        setUploading(false);
      }

      const userText = prompt || "I've uploaded a PDF. Please summarize it.";
      nextMessages = [...messages, { role: "user", content: userText }];
      setMessages(nextMessages);

      const result = await api(`/threads/${threadId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });
      setMessages([...nextMessages, { role: "assistant", content: result.answer }]);

      const refreshed = await api("/threads");
      setThreads(refreshed);
    } catch (reason) {
      setUploading(false);
      setError(reason.message);
      if (nextMessages !== messages) setMessages(nextMessages);
    } finally {
      setBusy(false);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  const isEmpty = messages.length === 0;
  const themePopover = (
    <div className="theme-popover" role="dialog" aria-label="Choose a theme">
      <div className="theme-popover-heading">Theme</div>
      <div className="theme-options" role="radiogroup" aria-label="Color themes">
        {themes.map((option) => (
          <button
            key={option.id}
            className={`theme-option ${theme === option.id ? "theme-option-selected" : ""}`}
            role="radio"
            aria-checked={theme === option.id}
            onClick={() => {
              setTheme(option.id);
              setThemeOpen(false);
            }}
          >
            <span className={`theme-swatch swatch-${option.id}`} aria-hidden="true">
              <i /><i /><i />
            </span>
            <span className="theme-option-copy">
              <strong>{option.name}</strong>
              <span>{option.description}</span>
            </span>
            {theme === option.id && <Check className="theme-check" size={19} strokeWidth={2.6} />}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="app-shell">
      {sidebarOpen && <button className="mobile-scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><Sparkles size={17} strokeWidth={2.3} /></div>
          <span className="brand-name">AI Chatbot</span>
          <button className="icon-button sidebar-close" aria-label="Close menu" onClick={() => setSidebarOpen(false)}><X size={18} /></button>
        </div>

        <button className="new-chat" onClick={newThread}>
          <Plus size={17} /> <span>New conversation</span>
        </button>

        <div className="sidebar-label">YOUR WORKSPACE</div>
        <div className="sidebar-nav-item active"><MessageSquare size={16} /><span>Conversations</span></div>

        <div className="history-heading">
          <span>Recent</span><span className="history-count">{threads.length}</span>
        </div>
        <div className="thread-list">
          {threads.length === 0 ? (
            <div className="empty-history">Your conversations will appear here.</div>
          ) : threads.map((thread) => (
            <button
              className={`thread-item ${thread.id === threadId ? "selected" : ""}`}
              key={thread.id}
              title={thread.title}
              onClick={() => openThread(thread.id)}
            >
              <MessageSquare size={15} />
              <span>{thread.title}</span>
            </button>
          ))}
        </div>

        <div className="sidebar-actions">
          <a
            className="sidebar-action"
            href="https://github.com/vchbtu71"
            target="_blank"
            rel="noreferrer"
          >
            <svg className="github-mark" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
              <path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.56.1.76-.24.76-.54v-2.1c-3.1.67-3.76-1.32-3.76-1.32-.5-1.28-1.23-1.62-1.23-1.62-1.01-.69.08-.68.08-.68 1.12.08 1.7 1.15 1.7 1.15.99 1.69 2.6 1.2 3.24.92.1-.72.39-1.2.7-1.48-2.47-.28-5.07-1.24-5.07-5.5 0-1.22.44-2.22 1.15-3-.12-.28-.5-1.42.11-2.96 0 0 .94-.3 3.05 1.15a10.6 10.6 0 0 1 5.55 0c2.11-1.45 3.05-1.15 3.05-1.15.61 1.54.23 2.68.11 2.96.72.78 1.15 1.78 1.15 3 0 4.27-2.6 5.21-5.08 5.49.4.35.75 1.02.75 2.06v3.08c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z" />
            </svg>
            <span>GitHub</span>
          </a>
          <a
            className="sidebar-action"
            href="https://www.linkedin.com/in/vchbtu71/"
            target="_blank"
            rel="noreferrer"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
              <path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.13 1.44-2.13 2.94v5.67H9.35V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.26 2.37 4.26 5.46v6.28ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.72v20.56C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.72V1.72C24 .77 23.2 0 22.22 0Z" />
            </svg>
            <span>LinkedIn</span>
          </a>
          <a className="sidebar-action" href="/api/resume" download>
            <Download size={18} />
            <span>Download resume</span>
          </a>
        </div>

        <div className="sidebar-footer">
          <div className="avatar">V</div>
          <div className="profile-copy"><strong>Workspace</strong><span>Personal account</span></div>
          <span className="online-dot" />
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button menu-toggle" aria-label="Open menu" onClick={() => setSidebarOpen(true)}><Menu size={19} /></button>
          <div className="topbar-right">
            <span className="status-dot" /> AI assistant <span className="topbar-model">· Groq</span>
            <div className="theme-control" ref={themeMenuRef}>
              <button
                className={`theme-trigger ${themeOpen ? "theme-trigger-open" : ""}`}
                onClick={() => setThemeOpen((open) => !open)}
                aria-label="Choose theme"
                aria-expanded={themeOpen}
                aria-haspopup="dialog"
              >
                <Palette size={19} strokeWidth={1.8} />
              </button>
              {themeOpen && themePopover}
            </div>
          </div>
        </header>

        <section className={`conversation ${isEmpty ? "conversation-empty" : ""}`}>
          {isEmpty ? (
            <div className="welcome-block">
              <div className="welcome-eyebrow"><span className="eyebrow-line" /> YOUR AI WORKSPACE <span className="eyebrow-line" /></div>
              <h1>Ideas start<br /><span>with a question.</span></h1>
              <p className="welcome-copy">A clear space to think, explore, and get things done.<br />Bring a document or start with a simple question.</p>
              <div className="suggestion-grid">
                {suggestions.map((item, index) => (
                  <button className="suggestion-card" key={item.title} onClick={() => sendMessage(item.prompt)}>
                    <span className={`suggestion-icon icon-${index}`}><item.icon size={16} strokeWidth={1.9} /></span>
                    <span className="suggestion-title">{item.title}</span>
                    <span className="suggestion-go"><ArrowUpRight size={13} /></span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article className={`message-row ${message.role}`} key={`${index}-${message.role}`}>
                  <div className={`message-avatar ${message.role === "assistant" ? "assistant-avatar" : "user-avatar"}`}>
                    {message.role === "assistant" ? <Sparkles size={15} /> : "V"}
                  </div>
                  <div className="message-body">
                    <div className="message-author">{message.role === "assistant" ? "AI Chatbot" : "You"}</div>
                    {message.role === "assistant" ? (
                      <div className="message-content assistant-markdown">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {normalizeAssistantMarkdown(message.content)}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="message-content">{message.content}</div>
                    )}
                  </div>
                </article>
              ))}
              {busy && (
                <article className="message-row assistant">
                  <div className="message-avatar assistant-avatar"><Sparkles size={15} /></div>
                  <div className="message-body"><div className="message-author">AI Chatbot</div><div className="thinking"><span /><span /><span /><em>{uploading ? "Reading your document" : "Thinking"}</em></div></div>
                </article>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-area">
          {uploadedDocument && (
            <div className="document-pill"><FileText size={15} /><span>{uploadedDocument.filename}</span><span className="document-pages">{uploadedDocument.documents} pages</span><Check size={14} className="document-check" /></div>
          )}
          {file && (
            <div className="file-preview"><FileText size={15} /><span>{file.name}</span><button className="icon-button remove-file" aria-label="Remove file" onClick={() => { setFile(null); setFileInputKey((value) => value + 1); }}><X size={15} /></button></div>
          )}
          {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={15} /></button></div>}
          <div className="composer">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Message AI Chatbot…"
              rows={1}
              aria-label="Message AI Chatbot"
              disabled={busy}
            />
            <div className="composer-toolbar">
              <div className="composer-tools">
                <label className={`attach-button ${busy ? "disabled" : ""}`} title="Attach a PDF">
                  <Paperclip size={16} />
                  <input
                    key={fileInputKey}
                    type="file"
                    accept="application/pdf,.pdf"
                    disabled={busy}
                    onChange={(event) => setFile(event.target.files?.[0] || null)}
                  />
                </label>
                <span className="composer-hint">Attach a PDF to ask about it</span>
              </div>
              <button className="send-button" onClick={() => sendMessage()} disabled={busy || (!draft.trim() && !file)} aria-label="Send message">
                {busy ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={18} />}
              </button>
            </div>
          </div>
          <div className="composer-disclaimer">AI Chatbot can make mistakes. Verify important information.</div>
        </div>
        <button className="scroll-bottom" aria-label="Scroll to latest" onClick={() => bottomRef.current?.scrollIntoView({ behavior: "smooth" })}><ArrowDown size={15} /></button>
      </main>
    </div>
  );
}

export default App;
