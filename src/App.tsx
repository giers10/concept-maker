import { useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { markdownToHTML } from "./markdown/markdown";
import "./markdown/markdown-render.css";

type FileEntry = {
  kind: "file";
  path: string;
  name: string;
  type: string;
  size: string;
};

type UrlEntry = {
  kind: "url";
  url: string;
  name: string;
  type: "url";
  size: string;
};

type RowEntry = FileEntry | UrlEntry;

type RephraseVariant = {
  key: string;
  label: string;
  text: string;
};

type SessionSummary = {
  title: string;
  description: string;
  saved_at: number;
};

type PriorArtResult = {
  url: string;
  title?: string;
  score?: number;
  snippet?: string;
};

type PriorArtResponse = {
  queries: string[];
  lang?: string;
  results: PriorArtResult[];
};

async function runBackend<T>(action: string, payload: Record<string, unknown>): Promise<T> {
  return (await invoke("run_python_action", { action, payload })) as T;
}

async function openPath(path: string): Promise<void> {
  await invoke("open_path", { path });
}

function formatTime(ts: number): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function rowId(row: RowEntry): string {
  return row.kind === "file" ? `file:${row.path}` : `url:${row.url}`;
}

function pdfFileNameFromTitle(title: string, conceptText: string): string {
  const heading = conceptText.match(/^#\s+(.+)$/m)?.[1] ?? "";
  const rawName = (title || heading || "concept").trim();
  const safeName = rawName
    .replace(/\s+/g, "-")
    .replace(/[^a-zA-Z0-9._-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-_.]+|[-_.]+$/g, "");

  return `${safeName || "concept"}.pdf`;
}

function ensurePdfExtension(path: string): string {
  return /\.pdf$/i.test(path) ? path : `${path}.pdf`;
}

function fallbackIdeaText(title: string, description: string, concept: string): string {
  return [
    title.trim() ? `Title: ${title.trim()}` : "",
    description.trim() ? `Description: ${description.trim()}` : "",
    concept.trim(),
  ]
    .filter(Boolean)
    .join("\n\n");
}

function currentIdeaText(notes: string, title: string, description: string, concept: string): string {
  return notes.trim() ? notes : fallbackIdeaText(title, description, concept);
}

export default function App() {
  const [status, setStatusMessage] = useState("");
  const [statusVersion, setStatusVersion] = useState(0);
  const [statusVisible, setStatusVisible] = useState(false);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [websites, setWebsites] = useState<UrlEntry[]>([]);
  const [notes, setNotes] = useState("");
  const [rephraseVariants, setRephraseVariants] = useState<RephraseVariant[]>([]);
  const [rephraseSelected, setRephraseSelected] = useState<string | null>(null);
  const [concept, setConcept] = useState("");
  const [markdownPreview, setMarkdownPreview] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [ollamaHost, setOllamaHost] = useState("http://localhost:11434");
  const [ollamaModel, setOllamaModel] = useState("Select model...");
  const [searxUrl, setSearxUrl] = useState("http://localhost:8888");
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  const [sessionModalOpen, setSessionModalOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);

  const [priorModalOpen, setPriorModalOpen] = useState(false);
  const [priorData, setPriorData] = useState<PriorArtResponse | null>(null);
  const [websiteModalOpen, setWebsiteModalOpen] = useState(false);
  const [websiteInput, setWebsiteInput] = useState("");
  const [websiteError, setWebsiteError] = useState("");
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const menuActionRef = useRef<(action: string) => void>(() => undefined);

  const rows = useMemo<RowEntry[]>(() => [...files, ...websites], [files, websites]);
  const markdownPreviewHtml = useMemo(
    () => (markdownPreview ? markdownToHTML(concept) : ""),
    [concept, markdownPreview]
  );

  const setStatus = (message: string) => {
    setStatusMessage(message);
    setStatusVersion((value) => value + 1);
  };

  useEffect(() => {
    if (!status) {
      setStatusVisible(false);
      return;
    }

    setStatusVisible(true);
    const timer = window.setTimeout(() => {
      setStatusVisible(false);
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [status, statusVersion]);

  useEffect(() => {
    let mounted = true;
    const bootstrap = async () => {
      try {
        const settings = await runBackend<Record<string, string>>("load_settings", {});
        if (!mounted) return;
        if (settings.ollama_host) setOllamaHost(settings.ollama_host);
        if (settings.ollama_model) setOllamaModel(settings.ollama_model);
        if (settings.searx_url) setSearxUrl(settings.searx_url);
      } catch {
        // ignore
      }
      try {
        const list = await runBackend<string[]>("list_models", {});
        if (!mounted) return;
        setModels(list);
      } catch {
        // ignore
      }
    };
    void bootstrap();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      runBackend("save_settings", {
        settings: {
          ollama_host: ollamaHost,
          ollama_model: ollamaModel,
          searx_url: searxUrl,
        },
      }).catch(() => undefined);
    }, 400);
    return () => clearTimeout(timer);
  }, [ollamaHost, ollamaModel, searxUrl]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    const attach = async () => {
      try {
        const current = getCurrentWebview() as any;
        unlisten = await current.onDragDropEvent((event: any) => {
          if (event.payload?.type === "drop") {
            const paths = event.payload.paths ?? [];
            if (paths.length) {
              void addFilesByPath(paths, true);
            }
          }
        });
      } catch {
        // optional
      }
    };
    void attach();
    return () => {
      if (unlisten) unlisten();
    };
  }, [files]);

  const addFilesByPath = async (paths: string[], expandDirs: boolean) => {
    if (!paths.length) return;
    setStatus("Indexing files...");
    try {
      const stats = await runBackend<{ name: string; path: string; type: string; size: string }[]>(
        "stat_paths",
        { paths, expand_dirs: expandDirs }
      );
      setFiles((prev) => {
        const existing = new Map(prev.map((f) => [f.path, f]));
        stats.forEach((item) => {
          if (!existing.has(item.path)) {
            existing.set(item.path, {
              kind: "file",
              path: item.path,
              name: item.name,
              type: item.type,
              size: item.size,
            });
          }
        });
        return Array.from(existing.values());
      });
      setStatus(`Added ${stats.length} item(s)`);
    } catch (err) {
      setStatus("Failed to add files");
      console.error(err);
    }
  };

  const onAddFiles = async () => {
    const selection = await open({ multiple: true, title: "Select files" });
    if (!selection) return;
    const paths = Array.isArray(selection) ? selection : [selection];
    await addFilesByPath(paths as string[], false);
  };

  const onAddWebsite = () => {
    setWebsiteInput("");
    setWebsiteError("");
    setWebsiteModalOpen(true);
  };

  const onSubmitWebsite = () => {
    const trimmed = websiteInput.trim();
    if (!/^https?:\/\//i.test(trimmed)) {
      setWebsiteError("Enter a valid http(s) URL.");
      return;
    }
    if (websites.some((w) => w.url === trimmed)) {
      setWebsiteError("This website is already in the table.");
      return;
    }

    const name = trimmed.replace(/^https?:\/\//, "").slice(0, 60);
    setWebsites((prev) => [
      ...prev,
      {
        kind: "url",
        url: trimmed,
        name,
        type: "url",
        size: "web",
      },
    ]);
    setWebsiteModalOpen(false);
    setWebsiteInput("");
    setWebsiteError("");
    setStatus(`Added ${name}`);
  };

  const closeWebsiteModal = () => {
    setWebsiteModalOpen(false);
    setWebsiteInput("");
    setWebsiteError("");
  };

  const onRemoveRow = (row: RowEntry) => {
    const label = row.kind === "file" ? row.path : row.url;
    if (!window.confirm(`Remove "${row.name}" from Files & Websites?\n\n${label}`)) return;
    if (row.kind === "file") {
      setFiles((prev) => prev.filter((file) => file.path !== row.path));
    } else {
      setWebsites((prev) => prev.filter((website) => website.url !== row.url));
    }
    setStatus(`Removed ${row.name}`);
  };

  const onRephrase = async () => {
    if (!ollamaModel || ollamaModel === "Select model...") {
      window.alert("Please select a model first.");
      return;
    }
    const ideaText = currentIdeaText(notes, title, description, concept);
    if (!ideaText.trim()) {
      window.alert("Add notes or a concept to rephrase.");
      return;
    }
    setBusy((prev) => ({ ...prev, rephrase: true }));
    setStatus("Rephrasing...");
    try {
      const variants = await runBackend<RephraseVariant[]>("rephrase", {
        note: ideaText,
        ollama_host: ollamaHost,
        model: ollamaModel,
      });
      setRephraseVariants(variants);
      setRephraseSelected(variants[0]?.key ?? null);
      setStatus("Rephrase complete");
    } catch (err) {
      setStatus("Rephrase failed");
      console.error(err);
      window.alert("Rephrase failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, rephrase: false }));
    }
  };

  const onExtend = async () => {
    if (!ollamaModel || ollamaModel === "Select model...") {
      window.alert("Please select a model first.");
      return;
    }
    const ideaText = currentIdeaText(notes, title, description, concept);
    if (!ideaText.trim()) {
      window.alert("Add notes or a concept to extend.");
      return;
    }
    setBusy((prev) => ({ ...prev, extend: true }));
    setStatus("Extending...");
    try {
      const extension = await runBackend<string>("extend", {
        note: ideaText,
        ollama_host: ollamaHost,
        model: ollamaModel,
      });
      setNotes((prev) => (prev.trim() ? `${prev}\n${extension}` : extension));
      setStatus("Extension ready");
    } catch (err) {
      setStatus("Extend failed");
      console.error(err);
      window.alert("Extend failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, extend: false }));
    }
  };

  const onGenerateConcept = async () => {
    if (!ollamaModel || ollamaModel === "Select model...") {
      window.alert("Please select a model first.");
      return;
    }
    if (!notes.trim() && !files.length && !websites.length) {
      window.alert("Add files, websites, or notes first.");
      return;
    }
    setBusy((prev) => ({ ...prev, generate: true }));
    setStatus("Generating concept...");
    try {
      const result = await runBackend<{ concept: string; title: string; description: string }>(
        "generate_concept",
        {
          notes,
          files: files.map((f) => f.path),
          websites: websites.map((w) => w.url),
          ollama_host: ollamaHost,
          model: ollamaModel,
        }
      );
      setConcept(result.concept);
      setTitle(result.title);
      setDescription(result.description);
      setStatus("Concept generated");
    } catch (err) {
      setStatus("Generation failed");
      console.error(err);
      window.alert("Generation failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, generate: false }));
    }
  };

  const onPriorArt = async () => {
    if (!ollamaModel || ollamaModel === "Select model...") {
      window.alert("Please select a model first.");
      return;
    }
    const priorArtNotes = currentIdeaText(notes, title, description, concept);
    setBusy((prev) => ({ ...prev, prior: true }));
    setStatus("Searching prior art...");
    try {
      const data = await runBackend<PriorArtResponse>("prior_art", {
        notes: priorArtNotes,
        concept,
        title,
        description,
        files: files.map((f) => f.path),
        websites: websites.map((w) => w.url),
        ollama_host: ollamaHost,
        model: ollamaModel,
        searx_url: searxUrl,
      });
      setPriorData(data);
      setPriorModalOpen(true);
      setStatus("Prior art ready");
    } catch (err) {
      setStatus("Prior art failed");
      console.error(err);
      window.alert("Prior art search failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, prior: false }));
    }
  };

  const onExportPdf = async () => {
    if (!concept.trim()) {
      window.alert("Generate or paste concept text first.");
      return;
    }
    setBusy((prev) => ({ ...prev, preview: true }));
    setStatus("Choose where to save the PDF...");
    try {
      const selectedPath = await save({
        title: "Export PDF",
        defaultPath: pdfFileNameFromTitle(title, concept),
        filters: [{ name: "PDF", extensions: ["pdf"] }],
        canCreateDirectories: true,
      });
      if (!selectedPath) {
        setStatus("PDF export cancelled");
        return;
      }

      setStatus("Exporting PDF...");
      const result = await runBackend<{ ok: boolean; pdf_path: string; log_path: string }>("preview_pdf", {
        concept,
        title,
        files: files.map((f) => f.path),
        output_path: ensurePdfExtension(selectedPath),
      });
      if (result.ok) {
        setStatus("Opening exported PDF...");
        try {
          await openPath(result.pdf_path);
          setStatus("PDF exported and opened");
        } catch (openErr) {
          setStatus("PDF exported");
          console.error(openErr);
          window.alert(`PDF exported to:\n${result.pdf_path}\n\nCould not open it automatically.`);
        }
      } else {
        setStatus("Export failed");
        window.alert(`Export failed. Log: ${result.log_path}`);
      }
    } catch (err) {
      setStatus("Export failed");
      console.error(err);
      window.alert("Export failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, preview: false }));
    }
  };

  const onNewSession = () => {
    if (notes || concept || files.length || websites.length) {
      const ok = window.confirm("Clear the current session?");
      if (!ok) return;
    }
    setFiles([]);
    setWebsites([]);
    setNotes("");
    setRephraseVariants([]);
    setRephraseSelected(null);
    setConcept("");
    setTitle("");
    setDescription("");
  };

  const onSaveSession = async () => {
    if (!title.trim()) {
      window.alert("Please enter a Title before saving.");
      return;
    }
    try {
      const existing = await runBackend<SessionSummary[]>("list_sessions", {});
      const exists = existing.some((s) => s.title === title.trim());
      if (exists) {
        const ok = window.confirm(`A session titled "${title}" exists. Overwrite it?`);
        if (!ok) return;
      }
      await runBackend("save_session", {
        payload: {
          title,
          description,
          notes,
          concept,
          files: files.map((f) => ({ path: f.path })),
          websites: websites.map((w) => ({ url: w.url })),
          rephrase_variants: rephraseVariants,
          rephrase_selected_key: rephraseSelected,
        },
        allow_overwrite: true,
      });
      setStatus("Session saved");
    } catch (err) {
      console.error(err);
      window.alert("Failed to save session.");
    }
  };

  const onOpenSession = async () => {
    setSessionModalOpen(true);
    setSessionsLoading(true);
    setSessions([]);
    try {
      const list = await runBackend<SessionSummary[]>("list_sessions", {});
      setSessions(list.sort((a, b) => (b.saved_at || 0) - (a.saved_at || 0)));
    } catch (err) {
      console.error(err);
      setSessionModalOpen(false);
      window.alert("Failed to load sessions.");
    } finally {
      setSessionsLoading(false);
    }
  };

  const applySession = async (sessionTitle: string) => {
    try {
      const data = await runBackend<any>("load_session", { title: sessionTitle });
      if (!data) return;
      setTitle(data.title || "");
      setDescription(data.description || "");
      setNotes(data.notes || "");
      setConcept(data.concept || "");
      setFiles(
        (data.files || []).map((f: any) => ({
          kind: "file",
          path: f.path,
          name: f.path.split(/[\\/]/).pop() || f.path,
          type: (f.path.split(".").pop() || "file").toLowerCase(),
          size: "?",
        }))
      );
      setWebsites(
        (data.websites || []).map((w: any) => ({
          kind: "url",
          url: w.url,
          name: w.url.replace(/^https?:\/\//, "").slice(0, 60),
          type: "url",
          size: "web",
        }))
      );
      setRephraseVariants(data.rephrase_variants || []);
      setRephraseSelected(data.rephrase_selected_key || null);
      setSessionModalOpen(false);
      setStatus("Session loaded");
    } catch (err) {
      console.error(err);
      window.alert("Failed to open session.");
    }
  };

  useEffect(() => {
    menuActionRef.current = (action: string) => {
      switch (action) {
        case "file-new":
          onNewSession();
          break;
        case "file-open":
          void onOpenSession();
          break;
        case "file-save":
          void onSaveSession();
          break;
        case "settings-open":
          setSettingsModalOpen(true);
          break;
      }
    };
  });

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let disposed = false;

    listen<string>("app-menu-action", (event) => {
      menuActionRef.current(event.payload);
    })
      .then((cleanup) => {
        if (disposed) {
          cleanup();
        } else {
          unlisten = cleanup;
        }
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      if (unlisten) unlisten();
    };
  }, []);

  return (
    <div className="app">
      <div className="panes">
        <div className="column">
          <section className="panel notes-panel">
            <div className="panel-title">
              <h2>Notes / Thoughts</h2>
              <span>Scratchpad for raw ideas</span>
            </div>
            <textarea
              className="notes-area"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Drop in fragments, problem statements, or constraints..."
            />

            {rephraseVariants.length > 0 && (
              <div className="rephrase-panel">
                <strong style={{ fontSize: "12px", color: "var(--accent)" }}>Rephrase variants</strong>
                <div className="rephrase-list">
                  {rephraseVariants.map((variant) => (
                    <div
                      key={variant.key}
                      className={`rephrase-item ${rephraseSelected === variant.key ? "active" : ""}`}
                      onClick={() => {
                        setRephraseSelected(variant.key);
                        setNotes(variant.text);
                      }}
                    >
                      <div style={{ fontWeight: 600 }}>{variant.label}</div>
                      <div style={{ fontSize: "11px", color: "var(--subtle)" }}>
                        {variant.text.slice(0, 120)}{variant.text.length > 120 ? "..." : ""}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="controls">
              <button onClick={onRephrase} disabled={busy.rephrase}>Rephrase</button>
              <button onClick={onExtend} disabled={busy.extend}>Extend</button>
              <button onClick={onPriorArt} disabled={busy.prior}>Find Prior Art</button>
              <button className="primary" onClick={onGenerateConcept} disabled={busy.generate}>Generate Concept</button>
            </div>
          </section>

          <section className="panel files-panel">
            <div className="panel-title">
              <h2>Files & Websites</h2>
              <div className="panel-title-actions">
                <button onClick={onAddFiles}>Add Files</button>
                <button onClick={onAddWebsite}>Add URL</button>
              </div>
            </div>
            <div className="table">
              <div className="table-header">
                <div>Name</div>
                <div>Path</div>
                <div>Type</div>
                <div>Size</div>
                <div>Remove</div>
              </div>
              <div className="table-body">
                {rows.length === 0 && (
                  <div className="table-row" style={{ gridTemplateColumns: "1fr" }}>
                    <div style={{ color: "var(--muted)" }}>Drop files here or use Add Files / Add URL.</div>
                  </div>
                )}
                {rows.map((row) => (
                  <div key={rowId(row)} className="table-row">
                    <div>{row.name}</div>
                    <div>{row.kind === "file" ? row.path : row.url}</div>
                    <div>{row.type}</div>
                    <div>{row.size}</div>
                    <div>
                      <button
                        className="remove-row-button danger"
                        type="button"
                        onClick={() => onRemoveRow(row)}
                        aria-label={`Remove ${row.name}`}
                        title={`Remove ${row.name}`}
                      >
                        X
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>

        <div className="column">
          <section className="panel" style={{ flex: 1 }}>
            <div className="panel-title">
              <h2>{markdownPreview ? "Concept (Preview)" : "Concept (Editable)"}</h2>
              <span>{markdownPreview ? "Rendered Markdown" : "Refine and publish"}</span>
            </div>
            {markdownPreview ? (
              <div
                className="markdown-preview md-root"
                role="textbox"
                aria-readonly="true"
                tabIndex={0}
                dangerouslySetInnerHTML={{
                  __html:
                    markdownPreviewHtml ||
                    '<p class="md-paragraph markdown-preview__empty">No concept markdown to preview.</p>',
                }}
              />
            ) : (
              <>
                <div className="concept-meta">
                  <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title" />
                  <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description" />
                </div>
                <textarea
                  className="concept-editor"
                  value={concept}
                  onChange={(e) => setConcept(e.target.value)}
                  placeholder="Generated concept markdown appears here..."
                />
              </>
            )}
            <div className="controls concept-actions">
              <button onClick={onExportPdf} disabled={busy.preview}>Export PDF</button>
              <button onClick={() => setMarkdownPreview((value) => !value)}>
                {markdownPreview ? "Edit Markdown" : "Preview Markdown"}
              </button>
            </div>
          </section>
        </div>
      </div>

      {websiteModalOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeWebsiteModal();
            }
          }}
        >
          <form
            className="settings-window website-window"
            role="dialog"
            aria-modal="true"
            aria-labelledby="website-title"
            onSubmit={(event) => {
              event.preventDefault();
              onSubmitWebsite();
            }}
          >
            <div className="settings-header">
              <h3 id="website-title">Add URL</h3>
              <button className="ghost" type="button" onClick={closeWebsiteModal}>
                Close
              </button>
            </div>
            <div className="website-form">
              <label htmlFor="website-url">URL</label>
              <input
                id="website-url"
                autoFocus
                value={websiteInput}
                onChange={(event) => {
                  setWebsiteInput(event.target.value);
                  if (websiteError) setWebsiteError("");
                }}
                placeholder="https://example.com"
                type="url"
              />
              {websiteError && <div className="form-error">{websiteError}</div>}
              <div className="form-actions">
                <button type="button" onClick={closeWebsiteModal}>Cancel</button>
                <button className="primary" type="submit">Add URL</button>
              </div>
            </div>
          </form>
        </div>
      )}

      {sessionModalOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSessionModalOpen(false);
            }
          }}
        >
          <div
            className="settings-window session-window"
            role="dialog"
            aria-modal="true"
            aria-labelledby="session-title"
          >
            <div className="settings-header">
              <h3 id="session-title">Open Session</h3>
              <button className="ghost" onClick={() => setSessionModalOpen(false)}>
                Close
              </button>
            </div>
            {sessionsLoading && <p>Loading sessions...</p>}
            {!sessionsLoading && sessions.length === 0 && <p>No saved sessions yet.</p>}
            <div className="session-list">
              {!sessionsLoading &&
                sessions.map((s) => (
                  <button
                    key={s.title}
                    className="ghost session-item"
                    onClick={() => applySession(s.title)}
                  >
                    <strong>{s.title}</strong>
                    <div className="session-description">{s.description}</div>
                    <div className="session-time">{formatTime(s.saved_at)}</div>
                  </button>
                ))}
            </div>
          </div>
        </div>
      )}

      {settingsModalOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSettingsModalOpen(false);
            }
          }}
        >
          <div
            className="settings-window"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <div className="settings-header">
              <h3 id="settings-title">Settings</h3>
              <button className="ghost" onClick={() => setSettingsModalOpen(false)}>
                Close
              </button>
            </div>

            <div className="settings-grid">
              <label htmlFor="settings-ollama-host">Ollama host</label>
              <input
                id="settings-ollama-host"
                value={ollamaHost}
                onChange={(e) => setOllamaHost(e.target.value)}
              />

              <label htmlFor="settings-model">Model</label>
              <select
                id="settings-model"
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
              >
                <option value="Select model...">Select model...</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              <label htmlFor="settings-searx-url">SearXNG URL</label>
              <input
                id="settings-searx-url"
                value={searxUrl}
                onChange={(e) => setSearxUrl(e.target.value)}
              />
            </div>
          </div>
        </div>
      )}

      {priorModalOpen && priorData && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(69, 10, 10, 0.28)",
            display: "grid",
            placeItems: "center",
            zIndex: 10,
          }}
        >
          <div
            style={{
              background: "var(--panel)",
              border: "1px solid var(--line)",
              padding: "20px",
              borderRadius: "6px",
              width: "min(680px, 92vw)",
              maxHeight: "85vh",
              overflow: "auto",
              boxShadow: "var(--dialog-shadow)",
            }}
          >
            <h3 style={{ marginTop: 0 }}>Prior Art Results</h3>
            <div style={{ marginBottom: "12px" }}>
              <strong>Queries</strong>
              <ul style={{ marginTop: "6px" }}>
                {priorData.queries.map((q) => (
                  <li key={q} style={{ fontSize: "12px", color: "var(--ink)" }}>
                    {q}
                  </li>
                ))}
              </ul>
            </div>
            {priorData.results.length === 0 && <p>No results found.</p>}
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {priorData.results.map((r) => (
                <div key={r.url} style={{ border: "1px solid var(--line)", padding: "12px", borderRadius: "4px", background: "var(--surface)" }}>
                  <div style={{ fontWeight: 600, marginBottom: "4px" }}>{r.title || r.url}</div>
                  <div style={{ fontSize: "12px", color: "var(--muted)" }}>{r.url}</div>
                  {r.snippet && (
                    <div style={{ marginTop: "6px", fontSize: "12px", color: "var(--ink)" }}>{r.snippet}</div>
                  )}
                  {typeof r.score === "number" && (
                    <div style={{ marginTop: "4px", fontSize: "11px", color: "var(--subtle)" }}>Score: {r.score.toFixed(3)}</div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ marginTop: "16px", textAlign: "right" }}>
              <button onClick={() => setPriorModalOpen(false)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {statusVisible && status && (
        <div className="status-toast" role="status" aria-live="polite">
          {status}
        </div>
      )}
    </div>
  );
}
