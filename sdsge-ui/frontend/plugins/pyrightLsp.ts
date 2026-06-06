import { spawn, execSync } from "child_process";
import { createRequire } from "module";
import type { Plugin } from "vite";
import { WebSocketServer } from "ws";
import type { WebSocket } from "ws";

const _require = createRequire(import.meta.url);

function findPyrightLangServer(): string {
  try {
    return _require.resolve("pyright/dist/pyright-langserver.js");
  } catch {
    throw new Error("pyright is not installed. Run: npm install -D pyright");
  }
}

function detectPythonPath(): string {
  const candidates =
    process.platform === "win32"
      ? ["python", "python3", "py"]
      : ["python3", "python"];
  for (const cmd of candidates) {
    try {
      const result = execSync(
        `${cmd} -c "import sys; print(sys.executable)"`,
        { encoding: "utf-8", timeout: 5000, stdio: ["ignore", "pipe", "ignore"] },
      ).trim();
      if (result) return result;
    } catch {
      /* try next */
    }
  }
  return "";
}

export function pyrightLspPlugin(): Plugin {
  let pyrightPath: string;
  const pythonPath = detectPythonPath();

  return {
    name: "sdsge-pyright-lsp",

    configureServer(server) {
      try {
        pyrightPath = findPyrightLangServer();
      } catch (err) {
        console.warn("[pyright-lsp]", (err as Error).message);
        return;
      }

      if (pythonPath) {
        console.info(`[pyright-lsp] using Python at ${pythonPath}`);
      } else {
        console.warn("[pyright-lsp] could not detect Python; import completions may be limited");
      }

      // Serve LSP config so the browser client can pass pythonPath to Pyright
      server.middlewares.use("/_lsp/config", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.end(JSON.stringify({ pythonPath }));
      });

      const wss = new WebSocketServer({ noServer: true });

      server.httpServer?.on("upgrade", (request, socket, head) => {
        if (request.url === "/_lsp/python") {
          wss.handleUpgrade(
            request,
            socket as Parameters<typeof wss.handleUpgrade>[1],
            head,
            (ws) => {
              wss.emit("connection", ws, request);
            },
          );
        }
      });

      wss.on("connection", (ws: WebSocket) => {
        const pyright = spawn("node", [pyrightPath, "--stdio"], {
          stdio: ["pipe", "pipe", "inherit"],
        });

        let buf = Buffer.alloc(0);

        pyright.stdout.on("data", (chunk: Buffer) => {
          buf = Buffer.concat([buf, chunk]);
          while (true) {
            const sep = buf.indexOf("\r\n\r\n");
            if (sep === -1) break;
            const hdr = buf.slice(0, sep).toString("ascii");
            const m = /Content-Length:\s*(\d+)/i.exec(hdr);
            if (!m) {
              buf = buf.slice(sep + 4);
              break;
            }
            const len = parseInt(m[1], 10);
            const start = sep + 4;
            if (buf.length < start + len) break;
            const msg = buf.slice(start, start + len).toString("utf-8");
            buf = buf.slice(start + len);
            if (ws.readyState === 1 /* OPEN */) {
              ws.send(msg);
            }
          }
        });

        ws.on("message", (data: Buffer) => {
          const str = typeof data === "string" ? data : data.toString("utf-8");
          const bytes = Buffer.from(str, "utf-8");
          pyright.stdin.write(`Content-Length: ${bytes.length}\r\n\r\n`);
          pyright.stdin.write(bytes);
        });

        ws.on("close", () => {
          pyright.kill();
        });

        pyright.on("exit", (code) => {
          if (code !== null && code !== 0) {
            console.warn("[pyright-lsp] pyright-langserver exited with code", code);
          }
          if (ws.readyState === 1 /* OPEN */) ws.close();
        });

        pyright.on("error", (err) => {
          console.error(
            "[pyright-lsp] failed to spawn pyright-langserver:",
            err.message,
          );
          if (ws.readyState === 1 /* OPEN */) ws.close();
        });
      });
    },
  };
}
