import { ChevronDown, ChevronRight, GripVertical } from "lucide-react";
import { useState } from "react";
import type { CSSProperties, DragEvent, PointerEvent, ReactNode } from "react";

export interface PanelDef {
  id: string;
  title: string;
  badge?: ReactNode;
  content: ReactNode;
  headerActions?: ReactNode;
  defaultHeight?: number;
  noPadding?: boolean;
  scrollable?: boolean;
}

type LayoutDirection = "horizontal" | "vertical";
type DropPlacement = "top" | "right" | "bottom" | "left";

export function PanelWorkspace({
  panels,
  defaultLayout = "vertical",
  defaultSplit = 50,
  fillHeight = false,
  onFoldChange,
}: {
  panels: PanelDef[];
  defaultLayout?: LayoutDirection;
  defaultSplit?: number;
  fillHeight?: boolean;
  onFoldChange?: (folded: Record<string, boolean>) => void;
}) {
  const [order, setOrder] = useState<string[]>(() => panels.map((p) => p.id));
  const [folded, setFolded] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(panels.map((p) => [p.id, false])),
  );
  const [panelHeights, setPanelHeights] = useState<Record<string, number>>(() =>
    Object.fromEntries(panels.map((p) => [p.id, p.defaultHeight ?? 470])),
  );
  const [split, setSplit] = useState(defaultSplit);
  const [layout, setLayout] = useState<LayoutDirection>(defaultLayout);
  const [dragged, setDragged] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<{
    panel: string;
    placement: DropPlacement;
  } | null>(null);

  const byId = Object.fromEntries(panels.map((p) => [p.id, p]));
  const multi = order.length >= 2;

  function dragOverPanel(target: string, event: DragEvent<HTMLElement>) {
    event.preventDefault();
    if (dragged === null || dragged === target) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const distances: Array<[DropPlacement, number]> = [
      ["top", y],
      ["right", rect.width - x],
      ["bottom", rect.height - y],
      ["left", x],
    ];
    const placement = distances.reduce((a, b) => (b[1] < a[1] ? b : a))[0];
    setDropTarget((cur) =>
      cur?.panel === target && cur.placement === placement
        ? cur
        : { panel: target, placement },
    );
  }

  function dropPanel(target: string) {
    if (!dragged || dragged === target || !dropTarget) return;
    const draggedFirst =
      dropTarget.placement === "top" || dropTarget.placement === "left";
    const newLayout: LayoutDirection =
      dropTarget.placement === "top" || dropTarget.placement === "bottom"
        ? "vertical"
        : "horizontal";
    const without = order.filter((id) => id !== dragged);
    const ti = without.indexOf(target);
    without.splice(draggedFirst ? ti : ti + 1, 0, dragged);
    setOrder(without);
    if (newLayout !== layout) {
      setLayout(newLayout);
      setSplit(50);
    }
    setDragged(null);
    setDropTarget(null);
  }

  function startSplitResize(event: PointerEvent<HTMLDivElement>) {
    const ws = (event.currentTarget as HTMLElement).closest<HTMLElement>(
      ".output-workspace",
    );
    if (!ws) return;
    const rect = ws.getBoundingClientRect();
    const sx = event.clientX;
    const sy = event.clientY;
    const s0 = split;

    function move(e: globalThis.PointerEvent) {
      if (layout === "horizontal") {
        setSplit(Math.min(75, Math.max(25, s0 + ((e.clientX - sx) / rect.width) * 100)));
      } else {
        setSplit(Math.min(80, Math.max(20, s0 + ((e.clientY - sy) / rect.height) * 100)));
      }
    }
    function stop() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  function startHeightResize(id: string, event: PointerEvent<HTMLDivElement>) {
    const sy = event.clientY;
    const h0 = panelHeights[id] ?? 470;
    function move(e: globalThis.PointerEvent) {
      setPanelHeights((cur) => ({ ...cur, [id]: Math.max(260, h0 + e.clientY - sy) }));
    }
    function stop() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  function slotStyle(id: string, index: number): CSSProperties | undefined {
    if (layout !== "vertical") return undefined;
    const others = order.filter((oid) => oid !== id);
    if (folded[id]) return { flex: "0 0 auto" };
    if (others.some((oid) => folded[oid])) return { flex: 1, minHeight: 0 };
    if (order.length === 2) {
      return { flex: index === 0 ? split : 100 - split, minHeight: 0 };
    }
    return { flex: 1, minHeight: 0 };
  }

  function panelStyle(id: string): CSSProperties | undefined {
    if (folded[id]) return undefined;
    if (layout === "vertical") return { flex: 1 };
    if (fillHeight) return undefined;
    return { height: panelHeights[id] };
  }

  function toggleFold(id: string) {
    setFolded((current) => {
      const next = { ...current, [id]: !current[id] };
      onFoldChange?.(next);
      return next;
    });
  }

  function horizontalTemplate(): CSSProperties | undefined {
    if (layout !== "horizontal" || !multi) return undefined;
    if (order.length === 2) {
      if (folded[order[0]]) {
        return { gridTemplateColumns: "42px 6px minmax(0, 1fr)" };
      }
      if (folded[order[1]]) {
        return { gridTemplateColumns: "minmax(0, 1fr) 6px 42px" };
      }
    }
    return { gridTemplateColumns: `${split}fr 6px ${100 - split}fr` };
  }

  function bodyClass(def: PanelDef) {
    return [
      "output-panel-body",
      def.noPadding ? "no-padding" : "",
      def.scrollable ? "scrollable" : "",
    ]
      .filter(Boolean)
      .join(" ");
  }

  return (
    <section
      className={`output-workspace ${layout}${fillHeight ? " fill-height" : ""}`}
      style={horizontalTemplate()}
    >
      {order.map((id, index) => {
        const def = byId[id];
        if (!def) return null;
        const isFolded = folded[id];
        const prevId = index > 0 ? order[index - 1] : null;
        const showSplitter =
          multi && index > 0 && prevId !== null && !folded[prevId] && !isFolded;
        return (
          <div
            key={id}
            className={index > 0 ? "output-slot second" : "output-slot"}
            style={slotStyle(id, index)}
          >
            {showSplitter && (
              <div
                className="output-splitter"
                onPointerDown={order.length === 2 ? startSplitResize : undefined}
                style={order.length !== 2 ? { cursor: "default" } : undefined}
                title={order.length === 2 ? "Resize panels" : undefined}
              />
            )}
            <section
              className={`output-panel${isFolded ? " folded" : ""}`}
              style={panelStyle(id)}
              onDragOver={(e) => dragOverPanel(id, e)}
              onDragLeave={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget as Node)) {
                  setDropTarget(null);
                }
              }}
              onDrop={() => dropPanel(id)}
            >
              {dropTarget?.panel === id && (
                <div className={`panel-drop-indicator ${dropTarget.placement}`} />
              )}
              <header
                className="output-panel-header"
                draggable={multi}
                onDragStart={(e: DragEvent<HTMLElement>) => {
                  if (!multi) return;
                  e.dataTransfer.effectAllowed = "move";
                  setDragged(id);
                }}
                onDragEnd={() => {
                  setDragged(null);
                  setDropTarget(null);
                }}
              >
                <div>
                  {multi && <GripVertical size={15} />}
                  <strong>{def.title}</strong>
                  {def.badge != null && <span>{def.badge}</span>}
                </div>
                <div className="output-panel-actions">
                  {def.headerActions}
                  <button
                    className="icon-button"
                    onClick={() => toggleFold(id)}
                    title={isFolded ? `Expand ${def.title}` : `Fold ${def.title}`}
                  >
                    {isFolded ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                  </button>
                </div>
              </header>
              {!isFolded && (
                <>
                  <div className={bodyClass(def)}>{def.content}</div>
                  {!fillHeight && layout === "horizontal" && multi && (
                    <div
                      className="output-height-handle"
                      onPointerDown={(e) => startHeightResize(id, e)}
                      title={`Resize ${def.title} height`}
                    />
                  )}
                </>
              )}
            </section>
          </div>
        );
      })}
    </section>
  );
}
