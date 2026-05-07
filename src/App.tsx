import { useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { markdownToHTML } from "./markdown/markdown";
import "./markdown/markdown-render.css";

const IMAGE_PROMPT_PLACEHOLDER = "Generated image prompt will appear here.";

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
  const [imagePrompt, setImagePrompt] = useState(IMAGE_PROMPT_PLACEHOLDER);
  const [ollamaHost, setOllamaHost] = useState("http://localhost:11434");
  const [ollamaModel, setOllamaModel] = useState("Select model...");
  const [searxUrl, setSearxUrl] = useState("http://localhost:8888");
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  const [sessionModalOpen, setSessionModalOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);

  const [priorModalOpen, setPriorModalOpen] = useState(false);
  const [priorData, setPriorData] = useState<PriorArtResponse | null>(null);
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

  const onAddFolder = async () => {
    const selection = await open({ directory: true, title: "Select folder" });
    if (!selection) return;
    const paths = Array.isArray(selection) ? selection : [selection];
    await addFilesByPath(paths as string[], true);
  };

  const onAddWebsite = () => {
    const url = window.prompt("Enter a URL (starting with http:// or https://):");
    if (!url) return;
    const trimmed = url.trim();
    if (!/^https?:\/\//i.test(trimmed)) {
      window.alert("Please enter a valid http(s) URL.");
      return;
    }
    setWebsites((prev) => {
      if (prev.some((w) => w.url === trimmed)) return prev;
      const name = trimmed.replace(/^https?:\/\//, "").slice(0, 60);
      return [
        ...prev,
        {
          kind: "url",
          url: trimmed,
          name,
          type: "url",
          size: "web",
        },
      ];
    });
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
    if (!notes.trim()) {
      window.alert("Add some notes to rephrase.");
      return;
    }
    setBusy((prev) => ({ ...prev, rephrase: true }));
    setStatus("Rephrasing...");
    try {
      const variants = await runBackend<RephraseVariant[]>("rephrase", {
        note: notes,
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
    if (!notes.trim()) {
      window.alert("Add some notes to extend.");
      return;
    }
    setBusy((prev) => ({ ...prev, extend: true }));
    setStatus("Extending...");
    try {
      const extension = await runBackend<string>("extend", {
        note: notes,
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

  const onGenerateImagePrompt = async () => {
    if (!ollamaModel || ollamaModel === "Select model...") {
      window.alert("Please select a model first.");
      return;
    }
    const ideaText = [title ? `Title: ${title}` : "", description ? `Description: ${description}` : "", concept || notes]
      .filter(Boolean)
      .join("\n\n");
    if (!ideaText.trim()) {
      window.alert("Add concept text or notes first.");
      return;
    }
    setBusy((prev) => ({ ...prev, imagePrompt: true }));
    setStatus("Generating image prompt...");
    try {
      const promptText = await runBackend<string>("generate_image_prompt", {
        idea_text: ideaText,
        ollama_host: ollamaHost,
        model: ollamaModel,
      });
      setImagePrompt(promptText);
      setStatus("Image prompt ready");
    } catch (err) {
      setStatus("Image prompt failed");
      console.error(err);
      window.alert("Image prompt failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, imagePrompt: false }));
    }
  };

  const onGenerateImage = async () => {
    if (!imagePrompt.trim() || imagePrompt === IMAGE_PROMPT_PLACEHOLDER) {
      window.alert("Generate an image prompt first.");
      return;
    }
    const output = await open({ directory: true, title: "Select output folder" });
    if (!output) return;
    const outputDir = Array.isArray(output) ? output[0] : output;
    setBusy((prev) => ({ ...prev, imageGen: true }));
    setStatus("Generating image...");
    try {
      const result = await runBackend<{ output_path: string }>("generate_image", {
        prompt: imagePrompt,
        output_dir: outputDir,
        title,
      });
      setStatus(`Image saved: ${result.output_path}`);
      window.alert(`Image saved to:\n${result.output_path}`);
    } catch (err) {
      setStatus("Image generation failed");
      console.error(err);
      window.alert("Image generation failed. Check console for details.");
    } finally {
      setBusy((prev) => ({ ...prev, imageGen: false }));
    }
  };

  const onPriorArt = async () => {
    if (!ollamaModel || ollamaModel === "Select model...") {
      window.alert("Please select a model first.");
      return;
    }
    setBusy((prev) => ({ ...prev, prior: true }));
    setStatus("Searching prior art...");
    try {
      const data = await runBackend<PriorArtResponse>("prior_art", {
        notes,
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

  const onPreview = async () => {
    if (!concept.trim()) {
      window.alert("Generate or paste concept text first.");
      return;
    }
    setBusy((prev) => ({ ...prev, preview: true }));
    setStatus("Preparing preview...");
    try {
      const result = await runBackend<{ ok: boolean; pdf_path: string; log_path: string }>("preview_pdf", {
        concept,
        title,
        files: files.map((f) => f.path),
      });
      if (result.ok) {
        setStatus("Opening preview PDF...");
        try {
          await openPath(result.pdf_path);
          setStatus("Preview PDF opened");
        } catch (openErr) {
          setStatus("Preview PDF ready");
          console.error(openErr);
          window.alert(`Preview PDF saved to:\n${result.pdf_path}\n\nCould not open it automatically.`);
        }
      } else {
        setStatus("Preview failed");
        window.alert(`Preview failed. Log: ${result.log_path}`);
      }
    } catch (err) {
      setStatus("Preview failed");
      console.error(err);
      window.alert("Preview failed. Check console for details.");
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
    setSelectedRows(new Set());
    setNotes("");
    setRephraseVariants([]);
    setRephraseSelected(null);
    setConcept("");
    setTitle("");
    setDescription("");
    setImagePrompt(IMAGE_PROMPT_PLACEHOLDER);
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
          image_prompt: imagePrompt === IMAGE_PROMPT_PLACEHOLDER ? "" : imagePrompt,
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
    try {
      const list = await runBackend<SessionSummary[]>("list_sessions", {});
      setSessions(list.sort((a, b) => (b.saved_at || 0) - (a.saved_at || 0)));
      setSessionModalOpen(true);
    } catch (err) {
      console.error(err);
      window.alert("Failed to load sessions.");
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
      setImagePrompt(data.image_prompt || IMAGE_PROMPT_PLACEHOLDER);
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
          <section className="panel">
            <div className="panel-title">
              <h2>Files & Websites</h2>
              <span>Drag & drop files; add URLs</span>
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
                    <div style={{ color: "var(--muted)" }}>Drop files here or use the buttons below.</div>
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
            <div className="controls">
              <button onClick={onAddFiles}>Add Files</button>
              <button onClick={onAddFolder}>Add Folder</button>
              <button onClick={onAddWebsite}>Add Website</button>
            </div>
          </section>

          <section className="panel" style={{ flex: 1 }}>
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
              <button className="primary" onClick={onGenerateConcept} disabled={busy.generate}>Generate Concept</button>
              <button className="secondary" onClick={onPriorArt} disabled={busy.prior}>Find Prior Art</button>
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
            <div className="controls">
              <button onClick={onPreview} disabled={busy.preview}>Preview PDF</button>
              <button onClick={() => setMarkdownPreview((value) => !value)}>
                {markdownPreview ? "Edit Markdown" : "Preview Markdown"}
              </button>
              <button onClick={onGenerateImagePrompt} disabled={busy.imagePrompt}>Generate image prompt</button>
              <button onClick={onGenerateImage} disabled={busy.imageGen}>Generate Image</button>
            </div>
            <textarea
              className="image-prompt"
              value={imagePrompt}
              onChange={(e) => setImagePrompt(e.target.value)}
              placeholder={IMAGE_PROMPT_PLACEHOLDER}
            />
          </section>
        </div>
      </div>

      {sessionModalOpen && (
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
              width: "min(560px, 90vw)",
              maxHeight: "80vh",
              overflow: "auto",
              boxShadow: "var(--dialog-shadow)",
            }}
          >
            <h3 style={{ marginTop: 0 }}>Open Session</h3>
            {sessions.length === 0 && <p>No saved sessions yet.</p>}
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {sessions.map((s) => (
                <button
                  key={s.title}
                  className="ghost"
                  style={{ textAlign: "left", borderRadius: "4px" }}
                  onClick={() => applySession(s.title)}
                >
                  <strong>{s.title}</strong>
                  <div style={{ fontSize: "12px", color: "var(--muted)" }}>{s.description}</div>
                  <div style={{ fontSize: "11px", color: "var(--subtle)" }}>{formatTime(s.saved_at)}</div>
                </button>
              ))}
            </div>
            <div style={{ marginTop: "16px", textAlign: "right" }}>
              <button onClick={() => setSessionModalOpen(false)}>Close</button>
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
