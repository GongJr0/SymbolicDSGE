import { Copy, Download, Maximize2, X } from "lucide-react";
import { useState } from "react";
import type { FigureResult } from "./types";

function FigureCard({
  fig,
  onFullscreen,
}: {
  fig: FigureResult;
  onFullscreen: (fig: FigureResult) => void;
}) {
  function downloadFigure() {
    if (!fig.image_b64) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${fig.image_b64}`;
    link.download = `${fig.name}.png`;
    link.click();
  }

  async function copyFigure() {
    if (!fig.image_b64) return;
    try {
      const blob = await fetch(`data:image/png;base64,${fig.image_b64}`).then(
        (r) => r.blob(),
      );
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": blob }),
      ]);
    } catch {
      /* Clipboard API unavailable */
    }
  }

  if (fig.error) {
    return (
      <div className="figure-card figure-card-error">
        <div className="figure-card-toolbar">
          <span className="figure-card-name">{fig.name === "__error__" ? "Error" : fig.name}</span>
        </div>
        <div className="figure-card-err-body">
          <span className="status error">{fig.error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="figure-card">
      <div className="figure-card-toolbar">
        <span className="figure-card-name">{fig.name}</span>
        <button
          className="icon-button"
          onClick={() => onFullscreen(fig)}
          title="Fullscreen"
        >
          <Maximize2 size={13} />
        </button>
        <button
          className="icon-button"
          onClick={() => void copyFigure()}
          title="Copy to clipboard"
        >
          <Copy size={13} />
        </button>
        <button
          className="icon-button"
          onClick={downloadFigure}
          title="Download PNG"
        >
          <Download size={13} />
        </button>
      </div>
      <img
        src={`data:image/png;base64,${fig.image_b64}`}
        alt={fig.name}
        className="figure-img"
      />
    </div>
  );
}

function FigureModal({
  fig,
  onClose,
}: {
  fig: FigureResult;
  onClose: () => void;
}) {
  function downloadFigure() {
    if (!fig.image_b64) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${fig.image_b64}`;
    link.download = `${fig.name}.png`;
    link.click();
  }

  async function copyFigure() {
    if (!fig.image_b64) return;
    try {
      const blob = await fetch(`data:image/png;base64,${fig.image_b64}`).then(
        (r) => r.blob(),
      );
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": blob }),
      ]);
    } catch {
      /* Clipboard API unavailable */
    }
  }

  return (
    <div
      className="figure-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="figure-modal">
        <div className="figure-modal-toolbar">
          <span className="figure-card-name">{fig.name}</span>
          <button
            className="icon-button"
            onClick={() => void copyFigure()}
            title="Copy to clipboard"
          >
            <Copy size={14} />
          </button>
          <button
            className="icon-button"
            onClick={downloadFigure}
            title="Download PNG"
          >
            <Download size={14} />
          </button>
          <button className="icon-button" onClick={onClose} title="Close">
            <X size={14} />
          </button>
        </div>
        <div className="figure-modal-body">
          <img
            src={`data:image/png;base64,${fig.image_b64}`}
            alt={fig.name}
            className="figure-modal-img"
          />
        </div>
      </div>
    </div>
  );
}

export function FiguresPanel({ figures }: { figures: FigureResult[] }) {
  const [fullscreen, setFullscreen] = useState<FigureResult | null>(null);

  const visibleFigures = figures.filter((f) => f.name !== "__error__" || f.error);
  const hasFigures = visibleFigures.some((f) => f.image_b64);

  return (
    <>
      {fullscreen && (
        <FigureModal fig={fullscreen} onClose={() => setFullscreen(null)} />
      )}
      {!hasFigures && visibleFigures.length === 0 ? (
        <div className="figures-empty">
          <span className="muted">
            No figures yet — submit a plot function and run a simulation.
          </span>
        </div>
      ) : (
        <div className="figures-grid">
          {visibleFigures.map((fig, i) => (
            <FigureCard
              key={fig.name === "__error__" ? `__error__${i}` : fig.name}
              fig={fig}
              onFullscreen={setFullscreen}
            />
          ))}
          {!hasFigures && (
            <div className="figures-empty figures-empty-inline">
              <span className="muted">
                No figures yet — submit a plot function and run a simulation.
              </span>
            </div>
          )}
        </div>
      )}
    </>
  );
}
