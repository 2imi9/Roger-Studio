import { useEffect, useState } from "react";
import {
  createProject,
  deleteProject,
  readProject,
  searchProjects,
  updateProject,
  type ProjectRead,
} from "../api/client";
import { GuidedTour, type TourStep } from "./GuidedTour";

/**
 * Project menu — New / Save / Save As / Open / Rename / Delete.
 *
 * Design choice (see docs/RESEARCH-UX-IMPROVEMENTS.md Theme 1 +
 * OlmoEarth Studio API mapping): the smallest thing that unblocks
 * multi-day research is a Project resource that bundles the whole
 * session. Rather than building a proper state manager, we leverage
 * the fact that most of the app already reads/writes `sessionStorage`
 * (imagery layers, label features, custom tags, polygon, OE subtab,
 * last-visited chat, etc.). Save dumps every `geoenv.*` / `roger.*`
 * key into the project's state blob; Open restores those keys and
 * reloads the page so every component re-hydrates from the new state.
 *
 * Tradeoff: the "reload to load" step drops any unsaved in-memory
 * state (currently-running inference, open dropdown). That's fine for
 * MVP — "open a saved project" is a deliberate context switch. If it
 * becomes annoying we can plumb state-restore hooks into the App
 * tree instead of reloading.
 */

const SS_CURRENT_ID = "roger.project.currentId";
const SS_CURRENT_NAME = "roger.project.currentName";
const STATE_KEY_PREFIXES = ["geoenv.", "roger."];
// Keys that carry the identity of the current project itself — exclude
// them from the save payload so restoring a project doesn't write its
// own id into the state blob recursively.
const EXCLUDE_KEYS = new Set<string>([SS_CURRENT_ID, SS_CURRENT_NAME]);
// Keys that might contain SECRETS (API keys, HF tokens). These live in
// sessionStorage intentionally so they vanish at tab close and never
// touch disk — round-tripping them through a persisted Project blob
// would defeat that guarantee. Keep them per-browser-session only.
// Pattern match covers ".apiKey", ".hfToken", and any future ".token"
// or ".secret" additions without requiring this list to stay in sync.
const SECRET_KEY_PATTERNS = [/apikey$/i, /token$/i, /secret$/i, /password$/i];
function isSecretKey(key: string): boolean {
  return SECRET_KEY_PATTERNS.some((rx) => rx.test(key));
}

function snapshotSessionState(): Record<string, string> {
  const snap: Record<string, string> = {};
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (!key) continue;
    if (EXCLUDE_KEYS.has(key)) continue;
    if (isSecretKey(key)) continue;
    if (!STATE_KEY_PREFIXES.some((p) => key.startsWith(p))) continue;
    const val = sessionStorage.getItem(key);
    if (val !== null) snap[key] = val;
  }
  return snap;
}

function restoreSessionState(snap: Record<string, unknown>) {
  // Clear any existing geoenv./roger. keys first so stale values from
  // the previous session don't leak through. Exclude the current-project
  // tracker keys — those are set fresh below.
  const toRemove: string[] = [];
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (!key) continue;
    if (EXCLUDE_KEYS.has(key)) continue;
    if (STATE_KEY_PREFIXES.some((p) => key.startsWith(p))) toRemove.push(key);
  }
  toRemove.forEach((k) => sessionStorage.removeItem(k));
  // Now write everything from the saved state.
  for (const [k, v] of Object.entries(snap)) {
    if (typeof v === "string") sessionStorage.setItem(k, v);
  }
}

export function ProjectMenu() {
  const [open, setOpen] = useState(false);
  const [projects, setProjects] = useState<ProjectRead[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [currentId, setCurrentId] = useState<string | null>(() =>
    sessionStorage.getItem(SS_CURRENT_ID),
  );
  const [currentName, setCurrentName] = useState<string | null>(() =>
    sessionStorage.getItem(SS_CURRENT_NAME),
  );
  const [error, setError] = useState<string | null>(null);
  const [showOpenDialog, setShowOpenDialog] = useState(false);
  // Show the quickstart guide on first visit (no project saved yet AND no
  // sessionStorage flag saying we've dismissed it). Opt-in for returning
  // users via the menu item.
  const [showGuide, setShowGuide] = useState<boolean>(() => {
    try {
      if (sessionStorage.getItem("roger.guide.dismissed") === "1") return false;
      // Auto-open for first-time users — no current project + no prior dismissal.
      return !sessionStorage.getItem(SS_CURRENT_ID);
    } catch {
      return false;
    }
  });
  const dismissGuide = () => {
    setShowGuide(false);
    try { sessionStorage.setItem("roger.guide.dismissed", "1"); } catch { /* noop */ }
  };

  // Refresh the project list whenever the menu opens or the Open dialog
  // is shown — cheap enough (names + timestamps only, not state blobs
  // rendered until clicked).
  useEffect(() => {
    if (!open && !showOpenDialog) return;
    let cancelled = false;
    setLoadingList(true);
    searchProjects({ limit: 20 })
      .then((list) => { if (!cancelled) setProjects(list); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoadingList(false); });
    return () => { cancelled = true; };
  }, [open, showOpenDialog]);

  const persistCurrent = (id: string | null, name: string | null) => {
    setCurrentId(id);
    setCurrentName(name);
    if (id) sessionStorage.setItem(SS_CURRENT_ID, id);
    else sessionStorage.removeItem(SS_CURRENT_ID);
    if (name) sessionStorage.setItem(SS_CURRENT_NAME, name);
    else sessionStorage.removeItem(SS_CURRENT_NAME);
  };

  const handleNew = () => {
    if (!confirm("Start a new project? Unsaved work in the current session will be lost.")) return;
    // Nuke session state (same logic as restore but with empty payload)
    restoreSessionState({});
    persistCurrent(null, null);
    setOpen(false);
    location.reload();
  };

  const handleSave = async () => {
    setError(null);
    if (currentId) {
      // Save in place
      try {
        const updated = await updateProject(currentId, {
          name: currentName || "Untitled project",
          state: snapshotSessionState(),
        });
        persistCurrent(updated.id, updated.name);
        setOpen(false);
      } catch (e) {
        setError(String(e));
      }
    } else {
      // No current project — prompt for a name (Save As flow)
      await handleSaveAs();
    }
  };

  const handleSaveAs = async () => {
    const name = prompt("Save project as:", currentName || "New project");
    if (!name) return;
    setError(null);
    try {
      const created = await createProject({
        name,
        state: snapshotSessionState(),
      });
      persistCurrent(created.id, created.name);
      setOpen(false);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleOpen = async (p: ProjectRead) => {
    if (!confirm(`Open "${p.name}"? Unsaved work in the current session will be lost.`)) return;
    restoreSessionState(p.state);
    persistCurrent(p.id, p.name);
    setShowOpenDialog(false);
    setOpen(false);
    location.reload();
  };

  const handleRename = async () => {
    if (!currentId) return;
    const name = prompt("Rename project to:", currentName || "");
    if (!name || name === currentName) return;
    setError(null);
    try {
      const updated = await updateProject(currentId, {
        name,
        state: snapshotSessionState(),
      });
      persistCurrent(updated.id, updated.name);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleDelete = async (p: ProjectRead) => {
    if (!confirm(`Delete "${p.name}"? This cannot be undone.`)) return;
    try {
      await deleteProject(p.id);
      if (p.id === currentId) persistCurrent(null, null);
      setProjects((prev) => prev.filter((x) => x.id !== p.id));
    } catch (e) {
      setError(String(e));
    }
  };

  const displayLabel = currentName
    ? `Project · ${currentName}`
    : "Project · untitled";

  return (
    <div className="relative flex items-center gap-2" data-testid="project-menu">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="px-2.5 py-1 text-[11px] font-mono uppercase tracking-wider bg-geo-surface hover:bg-geo-elevated text-geo-muted hover:text-geo-text border border-geo-border rounded transition-colors cursor-pointer"
        title="Project menu — save / open / rename / delete"
      >
        {displayLabel}
      </button>
      <button
        type="button"
        onClick={() => setShowGuide(true)}
        data-testid="tour-launcher"
        title="Introduction Doc: walk through Roger Studio's key features"
        aria-label="Open Introduction Doc"
        className="px-2.5 py-1 text-[11px] font-mono uppercase tracking-wider bg-geo-accent/10 hover:bg-geo-accent hover:text-white text-geo-accent border border-geo-accent/40 rounded transition-colors cursor-pointer whitespace-nowrap"
      >
        Introduction Doc
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 z-40 w-48 bg-geo-bg border border-geo-border rounded-lg shadow-lg overflow-hidden">
          <MenuItem onClick={handleNew} label="New project" hint="Clear session + reload" />
          <MenuItem
            onClick={() => { setShowGuide(true); setOpen(false); }}
            label="Quickstart guide"
            hint="How to use Roger Studio"
          />
          <MenuItem
            onClick={handleSave}
            label={currentId ? "Save" : "Save as…"}
            hint={currentId ? "Persist current state" : "Name + create"}
          />
          {currentId && (
            <MenuItem onClick={handleSaveAs} label="Save as…" hint="Clone to a new name" />
          )}
          {currentId && (
            <MenuItem onClick={handleRename} label="Rename" hint="Change project name" />
          )}
          <MenuItem
            onClick={() => setShowOpenDialog(true)}
            label="Open…"
            hint={`${projects.length} saved`}
          />
          {error && (
            <div className="px-3 py-2 text-[10px] text-red-700 bg-red-50 border-t border-red-200">
              {error}
            </div>
          )}
        </div>
      )}

      {showOpenDialog && (
        <div
          className="fixed inset-0 z-50 bg-black/30 flex items-center justify-center"
          onClick={() => setShowOpenDialog(false)}
        >
          <div
            className="bg-geo-bg border border-geo-border rounded-xl shadow-xl w-[420px] max-h-[70vh] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-4 py-3 border-b border-geo-border flex items-center justify-between">
              <h3 className="m-0 text-sm font-semibold text-geo-text">Open project</h3>
              <button
                type="button"
                onClick={() => setShowOpenDialog(false)}
                className="text-geo-dim hover:text-geo-text text-base cursor-pointer"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {loadingList && (
                <div className="px-4 py-6 text-[12px] text-geo-muted italic">Loading…</div>
              )}
              {!loadingList && projects.length === 0 && (
                <div className="px-4 py-6 text-[12px] text-geo-muted italic">
                  No saved projects yet. Use Save on a session first.
                </div>
              )}
              {!loadingList &&
                projects.map((p) => (
                  <div
                    key={p.id}
                    className={`flex items-center justify-between px-4 py-2.5 border-b border-geo-border hover:bg-geo-elevated ${
                      p.id === currentId ? "bg-geo-accent/5" : ""
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => handleOpen(p)}
                      className="flex-1 text-left cursor-pointer"
                    >
                      <div className="text-[13px] font-medium text-geo-text">{p.name}</div>
                      <div className="text-[10px] text-geo-dim mt-0.5 font-mono">
                        {new Date(p.updated_at).toLocaleString()} · {p.id.slice(0, 8)}
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(p)}
                      className="ml-2 text-[10px] px-1.5 py-0.5 rounded text-red-700 hover:bg-red-100 cursor-pointer"
                      title="Delete project"
                    >
                      ×
                    </button>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {showGuide && <GuidedTour steps={TOUR_STEPS} onClose={dismissGuide} />}
    </div>
  );
}

// Shepherd-style guided tour. Each step spotlights a real UI element so
// the user learns by seeing where things live, not by reading a big wall
// of text up front. All selectors target always-visible sidebar / map
// chrome (tabs + Project button + Compare toggle) so no tab-switching
// is required mid-tour.
const TOUR_STEPS: TourStep[] = [
  {
    target: "[data-testid='roger-studio-brand']",
    placement: "bottom",
    title: "Welcome to Roger Studio",
    body: (
      <>
        Roger Studio is an Earth Observation copilot for remote sensing
        research. This 8-step tour walks through every tab and the
        matching map controls. Click Next and Back below, or press the
        arrow keys on your keyboard to step through. Press Esc at any
        time to close the tour.
      </>
    ),
  },
  {
    target: "[data-testid='project-menu']",
    placement: "bottom",
    title: "1. Project menu and Introduction Doc",
    body: (
      <>
        The <strong>Project button</strong> saves your current session
        (bbox, imagery layers, labels, chat history) as a persistent
        record you can reopen anytime. The <strong>Introduction Doc
        button</strong> right next to it reopens this tour. API keys
        always stay in your browser session and never touch the database.
      </>
    ),
  },
  {
    target: "[data-testid='tab-map']",
    placement: "bottom",
    title: "2. Map tab (sidebar)",
    body: (
      <>
        Hover the <strong>Map tab</strong> to open its submenu. Four
        subviews live here: <em>Labeling</em> has the Draw Rectangle and
        Draw Polygon tools plus manual annotation,{" "}
        <em>Sample Rasters</em> loads preset GeoTIFFs (Knoxville NDVI,
        Landsat, Wetland classification), <em>Sample Label</em> loads
        curated polygon sets (SF Parks, PA Karst, Solar Sites), and{" "}
        <em>Import Data</em> accepts your own GeoJSON or GeoTIFF
        uploads. The actual drawing happens on the map canvas in the
        next step.
      </>
    ),
  },
  {
    // Multi-target spotlight covers BOTH the top-left Draw controls and
    // the top-right Compare button. Using an array so the GuidedTour
    // component takes the union bbox of every matched element. Recenter
    // joins the highlight automatically whenever a selection exists
    // (its data-testid is inside the same draw-controls container).
    target: [
      "[data-testid='map-draw-controls']",
      "[data-testid='compare-mode-toggle']",
    ],
    placement: "bottom",
    title: "3. Map canvas controls",
    body: (
      <>
        On the map itself you have <strong>Draw Rectangle</strong> and{" "}
        <strong>Draw Polygon</strong> buttons on the top-left for defining
        your area, a <strong>Recenter</strong> button that fits the view
        back to your selection, and <strong>Compare</strong> on the
        top-right that opens a split A/B view for inspecting two rasters
        at once. These map controls pair with the Map tab sidebar: draw
        here, choose subview there. Large bboxes (above 50,000 km²)
        trigger a slowdown warning on the Analysis tab.
      </>
    ),
  },
  {
    target: "[data-testid='tab-olmoearth']",
    placement: "bottom",
    title: "4. OlmoEarth tab",
    body: (
      <>
        The <strong>OlmoEarth tab</strong> has three subtabs:{" "}
        <em>Encoder</em> (base weights), <em>Finetune head</em> (task
        heads: Mangrove, AWF, ForestLossDriver, EcosystemTypeMapping,
        LFMC), and <em>Dataset</em> (cached training corpora). Pick a
        model, click <span className="font-mono">Load</span> to cache
        weights, then <span className="font-mono">Run</span> to perform
        inference over your selected bbox. Results paint on the map as a
        classified or regression raster with a per-class legend.
      </>
    ),
  },
  {
    target: "[data-testid='tab-analysis']",
    placement: "bottom",
    title: "5. Analysis tab",
    body: (
      <>
        The <strong>Analysis tab</strong> summarizes every raster on the
        map. OlmoEarth inference layers expand into a per-class legend
        with scene metadata (S2 scene id, date, cloud cover) and a
        download-GeoTIFF button. Non-inference layers (Sample Rasters,
        uploads, STAC composites) show as a compact row with a viridis
        colormap swatch so you can see the color scheme at a glance.
      </>
    ),
  },
  {
    target: "[data-testid='tab-llm']",
    placement: "bottom",
    title: "6. LLM tab",
    body: (
      <>
        The <strong>LLM tab</strong> connects to five chat providers:
        Local Gemma, NVIDIA NIM, Claude, Gemini, and ChatGPT. All five
        share the same 11 geo-tools
        (<span className="font-mono">query_olmoearth</span>,{" "}
        <span className="font-mono">run_olmoearth_inference</span>,{" "}
        <span className="font-mono">query_ndvi_timeseries</span>, and
        more). Open the <em>Examples</em> subtab for copy-paste prompts
        that chain multiple tools into full analyses like mangrove extent
        mapping or forest-loss driver detection.
      </>
    ),
  },
  {
    // Spotlight unions the pop-out chip (visible only when on LLM tab,
    // not already floating) with the LLM tab button itself. If the
    // user isn't on LLM yet, only the tab highlights and the body
    // directs them to switch there; if they are, both light up so the
    // chip is findable inside the pane.
    target: [
      "[data-testid='llm-popout-toggle']",
      "[data-testid='tab-llm']",
    ],
    placement: "right",
    title: "7. Pop out the chat",
    body: (
      <>
        Inside the LLM tab there's a pop-out chip that detaches the chat
        into a floating, draggable panel. The pop-out is handy for
        multi-tool analysis sessions where you want to keep the chat
        visible while panning the map. The floating window remembers
        your last-visited provider, so switching to Settings or Examples
        will not blank out the active conversation.
      </>
    ),
  },
  {
    target: "[data-testid='project-menu']",
    placement: "bottom",
    title: "8. Save your work",
    body: (
      <>
        When you're happy with what's on screen, click{" "}
        <strong>Project · untitled</strong>, then choose{" "}
        <em>Save as</em>. Your bbox, imagery layers, labels, custom tags,
        and chat history bundle into a persistent record you can reopen
        anytime. Reopen this tour via the Introduction Doc button any
        time via the <em>Quickstart guide</em> entry in the project
        dropdown.
      </>
    ),
  },
];

function MenuItem({
  onClick, label, hint,
}: { onClick: () => void; label: string; hint?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left px-3 py-2 hover:bg-geo-elevated cursor-pointer border-b border-geo-border last:border-b-0"
    >
      <div className="text-[12px] font-medium text-geo-text">{label}</div>
      {hint && <div className="text-[10px] text-geo-dim mt-0.5">{hint}</div>}
    </button>
  );
}
