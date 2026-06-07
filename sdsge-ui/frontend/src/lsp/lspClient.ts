type JsonRpcMessage = {
  jsonrpc: "2.0";
  id?: number | string | null;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: unknown;
};

type NotificationHandler = (params: unknown) => void;

export class LspClient {
  private socket: WebSocket;
  private requestId = 0;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: unknown) => void }>();
  private notificationHandlers = new Map<string, NotificationHandler[]>();
  private ready: Promise<void>;

  constructor(url: string) {
    this.socket = new WebSocket(url);
    this.socket.addEventListener("message", (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as JsonRpcMessage;
        if (msg.id !== undefined && msg.id !== null && !msg.method) {
          const id = typeof msg.id === "number" ? msg.id : parseInt(String(msg.id), 10);
          const entry = this.pending.get(id);
          if (entry) {
            this.pending.delete(id);
            if (msg.error) entry.reject(msg.error);
            else entry.resolve(msg.result);
          }
        } else if (msg.method) {
          const handlers = this.notificationHandlers.get(msg.method) ?? [];
          for (const handler of handlers) handler(msg.params);
        }
      } catch {
        /* ignore malformed messages */
      }
    });

    this.ready = new Promise((resolve, reject) => {
      this.socket.addEventListener("open", () => resolve());
      this.socket.addEventListener("error", () => reject(new Error("WebSocket error")));
    });
  }

  onNotification(method: string, handler: NotificationHandler): void {
    if (!this.notificationHandlers.has(method)) {
      this.notificationHandlers.set(method, []);
    }
    this.notificationHandlers.get(method)!.push(handler);
  }

  async sendRequest(method: string, params?: unknown): Promise<unknown> {
    await this.ready;
    const id = ++this.requestId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
    });
  }

  sendNotification(method: string, params?: unknown): void {
    if (this.socket.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ jsonrpc: "2.0", method, params }));
  }

  get isOpen(): boolean {
    return this.socket.readyState === WebSocket.OPEN;
  }
}

let _globalClient: LspClient | null = null;

export function getOrCreateLspClient(): LspClient {
  if (_globalClient === null) {
    const wsUrl = `ws://${window.location.hostname}:${window.location.port}/_lsp/python`;
    _globalClient = new LspClient(wsUrl);
  }
  return _globalClient;
}
