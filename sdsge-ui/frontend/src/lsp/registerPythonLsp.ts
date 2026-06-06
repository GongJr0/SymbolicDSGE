import type * as Monaco from "monaco-editor";
import { getOrCreateLspClient } from "./lspClient";

// LSP CompletionItemKind → Monaco CompletionItemKind
const LSP_KIND_MAP: Record<number, Monaco.languages.CompletionItemKind> = {
  1:  1,  // Text
  2:  0,  // Method
  3:  0,  // Function
  4:  0,  // Constructor
  5:  3,  // Field
  6:  3,  // Variable
  7:  4,  // Class
  8:  7,  // Interface
  9:  8,  // Module
  10: 9,  // Property
  12: 11, // Value
  13: 12, // Enum
  14: 13, // Keyword
  15: 14, // Snippet
  16: 15, // Color
  17: 16, // File
  18: 17, // Reference
};

// LSP DiagnosticSeverity → Monaco MarkerSeverity
const LSP_SEVERITY_MAP: Record<number, Monaco.MarkerSeverity> = {
  1: 8, // Error
  2: 4, // Warning
  3: 2, // Info
  4: 1, // Hint
};

interface LspPosition {
  line: number;
  character: number;
}
interface LspRange {
  start: LspPosition;
  end: LspPosition;
}
interface LspDiagnostic {
  range: LspRange;
  severity?: number;
  message: string;
}
interface LspCompletionItem {
  label: string;
  kind?: number;
  detail?: string;
  documentation?: string | { kind: string; value: string };
  insertText?: string;
  insertTextFormat?: number;
}
interface LspCompletionList {
  isIncomplete: boolean;
  items: LspCompletionItem[];
}

// modelUri (sdsge-code://role-kind.py) → LSP fileUri (file:///sdsge-virtual/role-kind.py)
function monacoUriToLsp(uri: string): string {
  return uri.replace(/^sdsge-code:\/\//, "file:///sdsge-virtual/");
}

function lspRangeToMonaco(range: LspRange): Monaco.IRange {
  return {
    startLineNumber: range.start.line + 1,
    startColumn: range.start.character + 1,
    endLineNumber: range.end.line + 1,
    endColumn: range.end.character + 1,
  };
}

let initialized = false;
const openFiles = new Set<string>();

async function fetchLspConfig(): Promise<{ pythonPath: string }> {
  try {
    const res = await fetch("/_lsp/config");
    if (!res.ok) return { pythonPath: "" };
    return (await res.json()) as { pythonPath: string };
  } catch {
    return { pythonPath: "" };
  }
}

export async function registerPythonLsp(monaco: typeof Monaco): Promise<void> {
  if (initialized) return;
  initialized = true;

  let client: ReturnType<typeof getOrCreateLspClient>;
  try {
    client = getOrCreateLspClient();
  } catch {
    return;
  }

  const { pythonPath } = await fetchLspConfig();

  // Wait for connection then initialize
  try {
    await client.sendRequest("initialize", {
      processId: null,
      rootUri: "file:///sdsge-virtual",
      workspaceFolders: [{ uri: "file:///sdsge-virtual", name: "sdsge" }],
      capabilities: {
        textDocument: {
          synchronization: {
            dynamicRegistration: false,
            willSave: false,
            didSave: false,
            willSaveWaitUntil: false,
          },
          completion: {
            dynamicRegistration: false,
            completionItem: { snippetSupport: true, documentationFormat: ["plaintext"] },
          },
          hover: { dynamicRegistration: false, contentFormat: ["plaintext"] },
          publishDiagnostics: { relatedInformation: false },
        },
        workspace: { applyEdit: false },
      },
      initializationOptions: {
        typeCheckingMode: "basic",
        reportMissingImports: false,
        reportMissingModuleSource: false,
        ...(pythonPath ? { pythonPath } : {}),
      },
    });
    client.sendNotification("initialized", {});

    // Push configuration so Pyright picks up the Python interpreter
    if (pythonPath) {
      client.sendNotification("workspace/didChangeConfiguration", {
        settings: {
          python: { pythonPath },
          pyright: { typeCheckingMode: "basic" },
        },
      });
    }
  } catch {
    return; // LSP not available
  }

  // Receive diagnostics and apply to Monaco models
  client.onNotification("textDocument/publishDiagnostics", (params) => {
    const { uri, diagnostics } = params as {
      uri: string;
      diagnostics: LspDiagnostic[];
    };
    // Find Monaco model by matching the LSP URI back
    const monacoUri = uri.replace(/^file:\/\/\/sdsge-virtual\//, "sdsge-code://");
    const model = monaco.editor.getModels().find((m) => m.uri.toString() === monacoUri);
    if (!model) return;
    const markers: Monaco.editor.IMarkerData[] = diagnostics.map((d) => ({
      ...lspRangeToMonaco(d.range),
      message: d.message,
      severity: LSP_SEVERITY_MAP[d.severity ?? 1] ?? 8,
    }));
    monaco.editor.setModelMarkers(model, "pyright", markers);
  });

  // Register completion provider
  monaco.languages.registerCompletionItemProvider("python", {
    triggerCharacters: [".", " ", "("],
    async provideCompletionItems(model, position) {
      const uri = model.uri.toString();
      if (!uri.startsWith("sdsge-code://")) return { suggestions: [] };
      const lspUri = monacoUriToLsp(uri);

      try {
        const result = (await client.sendRequest("textDocument/completion", {
          textDocument: { uri: lspUri },
          position: { line: position.lineNumber - 1, character: position.column - 1 },
        })) as LspCompletionList | LspCompletionItem[] | null;

        const items: LspCompletionItem[] = result
          ? Array.isArray(result)
            ? result
            : result.items
          : [];

        const suggestions: Monaco.languages.CompletionItem[] = items.map((item) => {
          const docValue =
            typeof item.documentation === "string"
              ? item.documentation
              : item.documentation?.value ?? "";
          return {
            label: item.label,
            kind: LSP_KIND_MAP[item.kind ?? 1] ?? 0,
            detail: item.detail,
            documentation: docValue,
            insertText: item.insertText ?? item.label,
            insertTextRules:
              item.insertTextFormat === 2
                ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
                : undefined,
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column,
            },
          };
        });

        return { suggestions };
      } catch {
        return { suggestions: [] };
      }
    },
  });

  // Register hover provider
  monaco.languages.registerHoverProvider("python", {
    async provideHover(model, position) {
      const uri = model.uri.toString();
      if (!uri.startsWith("sdsge-code://")) return null;
      const lspUri = monacoUriToLsp(uri);

      try {
        const result = (await client.sendRequest("textDocument/hover", {
          textDocument: { uri: lspUri },
          position: { line: position.lineNumber - 1, character: position.column - 1 },
        })) as { contents: unknown; range?: LspRange } | null;

        if (!result?.contents) return null;

        const text =
          typeof result.contents === "string"
            ? result.contents
            : Array.isArray(result.contents)
              ? result.contents.map((c: unknown) => (typeof c === "string" ? c : (c as { value: string }).value ?? "")).join("\n")
              : (result.contents as { value: string }).value ?? "";

        return {
          contents: [{ value: `\`\`\`python\n${text}\n\`\`\`` }],
          range: result.range ? lspRangeToMonaco(result.range) : undefined,
        };
      } catch {
        return null;
      }
    },
  });

  // Sync open models to the LSP server
  function syncModel(model: Monaco.editor.ITextModel): void {
    const uri = model.uri.toString();
    if (!uri.startsWith("sdsge-code://")) return;
    const lspUri = monacoUriToLsp(uri);
    const text = model.getValue();

    if (!openFiles.has(lspUri)) {
      openFiles.add(lspUri);
      client.sendNotification("textDocument/didOpen", {
        textDocument: { uri: lspUri, languageId: "python", version: 1, text },
      });
    } else {
      client.sendNotification("textDocument/didChange", {
        textDocument: { uri: lspUri, version: model.getVersionId() },
        contentChanges: [{ text }],
      });
    }
  }

  // Sync all existing models
  for (const model of monaco.editor.getModels()) {
    if (model.getLanguageId() === "python") syncModel(model);
  }

  // Sync new models when created
  monaco.editor.onDidCreateModel((model) => {
    if (model.getLanguageId() === "python") syncModel(model);
  });

  // Sync on content change
  monaco.editor.onDidCreateModel((model) => {
    model.onDidChangeContent(() => {
      if (model.getLanguageId() === "python") syncModel(model);
    });
  });
}
