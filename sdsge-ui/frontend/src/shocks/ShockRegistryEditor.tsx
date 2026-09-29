// The shock panel, shared by every place a simulation's shocks are configured.
//
// The checklist is the target model's declared innovations (offered as options,
// never assumed); one entry is a free-form shock over the chosen subset. The
// kind is a property of the entry, not of the panel: an entry either draws from
// a family or supplies an array, which is the same split the spec carries and
// the same one `registryFromShocks` reads back. Client-side validation blocks an
// innovation already claimed by another entry, and nothing else about the family:
// the library decides which ones exist and which it draws univariate only. Path
// length is not checked here either; only the run knows `T`.

import { Check, Plus, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { ShockCatalog, ShockDistribution, ShockRegistryEntry } from "../types";
import { getShockCatalog } from "../api";
import { formatPathText, parsePathText } from "./registry";

// Display names only. The library has no business owning these, and a family it
// gains before anyone labels it here renders as its own code rather than as a
// blank row.
const DIST_LABEL: Record<string, string> = {
    norm: "Normal",
    t: "Student-t",
    uni: "Uniform",
    exp: "Exponential",
    gamma: "Gamma",
    beta: "Beta",
};

function distLabel(dist: string): string {
    return DIST_LABEL[dist] ?? dist;
}

// The fields the form offers per family, keyed by the name the library reads each
// parameter under. The panel's own list: the library requires these inside
// `validate_shock_family` as control flow, not as data a catalog could serve. A
// family missing here gets no extra fields and the library says what it wanted.
const DIST_PARAMS: Record<string, { key: string; label: string; value: string }[]> = {
    t: [{ key: "df", label: "Degrees of freedom", value: "5" }],
    gamma: [{ key: "a", label: "Shape (a)", value: "2" }],
    beta: [
        { key: "a", label: "Shape (a)", value: "2" },
        { key: "b", label: "Shape (b)", value: "5" },
    ],
};

function paramDefaults(dist: string): Record<string, string> {
    return Object.fromEntries(
        (DIST_PARAMS[dist] ?? []).map((field) => [field.key, field.value]),
    );
}

// One request per session however many panels mount, and the families outlive
// any one of them.
let catalogRequest: Promise<ShockCatalog> | null = null;

function shockFamilies(): Promise<ShockCatalog> {
    catalogRequest ??= getShockCatalog();
    return catalogRequest;
}

type EntryKind = ShockRegistryEntry["kind"];

export function ShockRegistryEditor({
    role,
    shockNames,
    entries,
    onChange,
}: {
    role: string;
    shockNames: string[];
    entries: ShockRegistryEntry[];
    onChange: (entries: ShockRegistryEntry[]) => void;
}) {
    const [kind, setKind] = useState<EntryKind>("drawn");
    const [selected, setSelected] = useState<string[]>([]);
    const [dist, setDist] = useState<ShockDistribution>("norm");
    const [families, setFamilies] = useState<string[]>([]);
    // Keyed by innovation, not positional: `commitEntry` reorders the selection
    // into the model's own order, and a positional list would hand each location
    // to the wrong shock.
    const [locs, setLocs] = useState<Record<string, string>>({});
    // Keyed by the library's own parameter names, as text while the form is open.
    const [params, setParams] = useState<Record<string, string>>(paramDefaults("norm"));
    const [seed, setSeed] = useState("");
    // The path as typed. It lives as text only while the form is open; an entry
    // holds the parsed matrix, which is what a grid editor would produce too.
    const [pathText, setPathText] = useState("");
    const [error, setError] = useState("");
    // Index of the entry being edited (null while composing a fresh one). Indices
    // are the spec's own, since both kinds are entries. The entry under edit is
    // excluded from the "used" set so its own innovations stay selectable, and
    // Save replaces it in place instead of appending.
    const [editIndex, setEditIndex] = useState<number | null>(null);

    useEffect(() => {
        let live = true;
        shockFamilies()
            .then((catalog) => {
                if (live) setFamilies(catalog.families);
            })
            .catch(() => {
                // The selector falls back to the current family below, so a failed
                // catalog leaves the form usable rather than empty.
            });
        return () => {
            live = false;
        };
    }, []);

    // Always offer what is selected, whether the catalog arrived or carries it: a
    // family the panel dropped from the list would be rewritten on the next edit.
    const distOptions = families.includes(dist) ? families : [dist, ...families];

    // Every entry claims its innovations, whichever kind it is.
    const usedNames = new Set(
        entries.flatMap((entry, index) => (index === editIndex ? [] : entry.target)),
    );
    const locFor = (name: string) => locs[name] ?? "0";
    const multivarJoint = kind === "drawn" && selected.length > 1;

    function resetForm() {
        setEditIndex(null);
        setKind("drawn");
        setSelected([]);
        setDist("norm");
        setLocs({});
        setParams(paramDefaults("norm"));
        setSeed("");
        setPathText("");
        setError("");
    }

    function toggleVar(name: string) {
        setError("");
        setSelected((current) =>
            current.includes(name)
                ? current.filter((item) => item !== name)
                : [...current, name],
        );
    }

    // Load an existing entry back into the form for editing.
    function startEdit(index: number) {
        const entry = entries[index];
        setEditIndex(index);
        setKind(entry.kind);
        setSelected(entry.target);
        setError("");
        if (entry.kind === "path") {
            setPathText(formatPathText(entry.path));
            return;
        }
        setDist(entry.dist);
        setLocs(
            Object.fromEntries(
                entry.target.map((name, position) => [name, String(entry.loc[position] ?? 0)]),
            ),
        );
        // The family's defaults first, so an entry authored without one of them
        // still offers a filled field rather than an empty one.
        setParams({
            ...paramDefaults(entry.dist),
            ...Object.fromEntries(
                Object.entries(entry.params).map(([key, value]) => [key, String(value)]),
            ),
        });
        setSeed(entry.seed === null ? "" : String(entry.seed));
    }

    // Build the entry this form describes, or report why it cannot be built. The
    // compiler in `shocksFromRegistry` refuses the same things, so this is the
    // early half of one rule rather than a second one.
    function entryFromForm(target: string[]): ShockRegistryEntry | null {
        if (kind === "path") {
            const path = parsePathText(pathText);
            if (path.length === 0) {
                setError("A supplied path needs at least one period.");
                return null;
            }
            const ragged = path.findIndex((row) => row.length !== target.length);
            if (ragged !== -1) {
                setError(
                    `Row ${ragged + 1} has ${path[ragged].length} value(s); ` +
                    `${target.length} selected.`,
                );
                return null;
            }
            if (path.some((row) => row.some((value) => !Number.isFinite(value)))) {
                setError("The path contains a value that is not a number.");
                return null;
            }
            return { kind: "path", target, path };
        }
        // One location per innovation, in the selection's order. An untouched box
        // is the form's own default of zero; anything else that is not a number is
        // an error rather than a silent zero.
        const loc: number[] = [];
        for (const name of target) {
            const raw = locFor(name).trim();
            const value = raw === "" ? 0 : Number(raw);
            if (!Number.isFinite(value)) {
                setError(`Location for '${name}' is not a number.`);
                return null;
            }
            loc.push(value);
        }
        // Blank is an error rather than a default here: the form seeded every field
        // the family needs, so an empty one was cleared on purpose.
        const numeric: Record<string, number> = {};
        for (const [key, raw] of Object.entries(params)) {
            const value = Number(raw.trim());
            if (raw.trim() === "" || !Number.isFinite(value)) {
                setError(`Parameter '${key}' is not a number.`);
                return null;
            }
            numeric[key] = value;
        }
        return {
            kind: "drawn",
            target,
            dist,
            loc,
            params: numeric,
            seed: seed.trim() === "" ? null : Number(seed),
        };
    }

    function commitEntry() {
        if (selected.length === 0) {
            setError("Select at least one innovation.");
            return;
        }
        const clash = selected.find((name) => usedNames.has(name));
        if (clash !== undefined) {
            setError(`'${clash}' is already used in another shock entry.`);
            return;
        }
        // Order the selection by the model's own shock order for a stable identity.
        const target = shockNames.filter((name) => selected.includes(name));
        const entry = entryFromForm(target);
        if (entry === null) return;
        onChange(
            editIndex === null
                ? [...entries, entry]
                : entries.map((current, index) => (index === editIndex ? entry : current)),
        );
        resetForm();
    }

    function removeEntry(index: number) {
        onChange(entries.filter((_, position) => position !== index));
        // Keep the edit target consistent with the shrunk list.
        if (editIndex === index) resetForm();
        else if (editIndex !== null && index < editIndex) setEditIndex(editIndex - 1);
    }

    return (
        <div className="mc-shock-registry">
            <div className="mc-shock-registry-head">
                <span className="mc-shock-registry-label">Shocks</span>
                <span className="mc-shock-registry-target">from {role}</span>
            </div>
            {entries.length > 0 ? (
                <ul className="mc-shock-list">
                    {entries.map((entry, index) => (
                        <li
                            key={`${entry.kind}:${index}`}
                            className={`mc-shock-entry${editIndex === index ? " editing" : ""}${entry.kind === "path" ? " mc-shock-entry-path" : ""
                                }`}
                        >
                            <button
                                className="mc-shock-entry-select"
                                title="Edit shock"
                                disabled={shockNames.length === 0}
                                onClick={() => startEdit(index)}
                            >
                                <div className="mc-shock-entry-body">
                                    <strong>
                                        {entry.target.length > 0 ? entry.target.join(", ") : "(unnamed)"}
                                        {entry.target.length > 1 && (
                                            <span className="mc-shock-badge">joint</span>
                                        )}
                                        {entry.kind === "path" && (
                                            <span className="mc-shock-badge">path</span>
                                        )}
                                    </strong>
                                    <span>{describeEntry(entry)}</span>
                                </div>
                            </button>
                            <button
                                className="icon-button"
                                title="Remove shock"
                                onClick={() => removeEntry(index)}
                            >
                                <Trash2 size={13} />
                            </button>
                        </li>
                    ))}
                </ul>
            ) : (
                <p className="mc-shock-empty">
                    No shocks configured; this simulation runs deterministically.
                </p>
            )}
            {shockNames.length === 0 ? (
                <p className="mc-shock-empty">
                    Load and solve the {role} model to choose its declared shocks.
                </p>
            ) : (
                <div className="mc-shock-form">
                    <div className="mc-shock-checklist">
                        {shockNames.map((name) => {
                            const used = usedNames.has(name);
                            return (
                                <label
                                    key={name}
                                    className={`mc-shock-check${used ? " used" : ""}`}
                                    title={used ? "Already used in another shock entry." : undefined}
                                >
                                    <input
                                        type="checkbox"
                                        checked={selected.includes(name)}
                                        disabled={used}
                                        onChange={() => toggleVar(name)}
                                    />
                                    <span>{name}</span>
                                </label>
                            );
                        })}
                    </div>
                    <div className="mc-shock-fields">
                        <label>
                            Source
                            <select
                                value={kind}
                                onChange={(event) => {
                                    setKind(event.target.value as EntryKind);
                                    setError("");
                                }}
                            >
                                <option value="drawn">Drawn</option>
                                <option value="path">Supplied path</option>
                            </select>
                        </label>
                        {kind === "drawn" && (
                            <>
                                <label>
                                    Distribution
                                    <select
                                        value={dist}
                                        onChange={(event) => {
                                            const next = event.target.value;
                                            setDist(next);
                                            // The previous family's parameters do not carry over; a
                                            // stale `df` on a gamma would travel to the library.
                                            setParams(paramDefaults(next));
                                            setError("");
                                        }}
                                    >
                                        {distOptions.map((family) => (
                                            <option key={family} value={family}>
                                                {distLabel(family)}
                                            </option>
                                        ))}
                                    </select>
                                </label>
                                {shockNames
                                    .filter((name) => selected.includes(name))
                                    .map((name) => (
                                        <label key={`loc:${name}`}>
                                            {selected.length > 1 ? `Location (${name})` : "Location"}
                                            <input
                                                type="number"
                                                value={locFor(name)}
                                                onChange={(event) =>
                                                    setLocs((current) => ({
                                                        ...current,
                                                        [name]: event.target.value,
                                                    }))
                                                }
                                            />
                                        </label>
                                    ))}
                                {(DIST_PARAMS[dist] ?? []).map((field) => (
                                    <label key={`param:${field.key}`}>
                                        {field.label}
                                        <input
                                            type="number"
                                            value={params[field.key] ?? ""}
                                            onChange={(event) =>
                                                setParams((current) => ({
                                                    ...current,
                                                    [field.key]: event.target.value,
                                                }))
                                            }
                                        />
                                    </label>
                                ))}
                                <label>
                                    Seed
                                    <input
                                        type="number"
                                        placeholder="none"
                                        value={seed}
                                        onChange={(event) => setSeed(event.target.value)}
                                    />
                                </label>
                            </>
                        )}
                    </div>
                    {kind === "path" && (
                        <label className="mc-shock-path">
                            Path
                            <textarea
                                className="shock-input"
                                value={pathText}
                                placeholder={selected.length > 1 ? "0 0\n1 0\n0 0" : "0\n1\n0"}
                                onChange={(event) => {
                                    setPathText(event.target.value);
                                    setError("");
                                }}
                            />
                        </label>
                    )}
                    {kind === "path" && (
                        <span className="mc-shock-hint">
                            One period per line, one value per selected innovation. The run
                            refuses a path that does not cover its horizon.
                        </span>
                    )}
                    {multivarJoint && (
                        <span className="mc-shock-hint">
                            This is one joint (multivar) shock over {selected.length}{" "}
                            innovations. Add a separate entry per innovation if you want them
                            independent or the distridution does not support multivariate draws.
                        </span>
                    )}
                    {error !== "" && <span className="status error mc-shock-error">{error}</span>}
                    <div className="mc-shock-actions">
                        <button
                            className="secondary mc-shock-add"
                            onClick={commitEntry}
                            disabled={selected.length === 0}
                        >
                            {editIndex === null ? <Plus size={13} /> : <Check size={13} />}
                            {editIndex === null ? "Add shock" : "Save shock"}
                        </button>
                        {editIndex !== null && (
                            <button className="secondary" onClick={resetForm}>
                                Cancel
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

function describeEntry(entry: ShockRegistryEntry): string {
    if (entry.kind === "path") {
        const periods = entry.path.length;
        return `Supplied over ${periods} ${periods === 1 ? "period" : "periods"}`;
    }
    const parts = [distLabel(entry.dist), `loc ${entry.loc.join(", ")}`];
    for (const [key, value] of Object.entries(entry.params)) {
        parts.push(`${key} ${value}`);
    }
    parts.push(entry.seed === null ? "seed none" : `seed ${entry.seed}`);
    return parts.join(", ");
}
