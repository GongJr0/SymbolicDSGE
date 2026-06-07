import type { MCPipelineResult, MCPipelineSpec } from "../types";

const DB_NAME = "symbolicdsge-ui";
const DB_VERSION = 1;
const STORE_NAME = "mc-workspace";
const WORKSPACE_KEY = "pipeline";
const RESULT_KEY = "result";

export interface MCPersistedWorkspace {
  version: 1;
  pipeline: MCPipelineSpec;
  positions: Record<string, { x: number; y: number }>;
  nRep: number;
  failFast: boolean;
}

export async function loadMCWorkspace(): Promise<MCPersistedWorkspace | null> {
  return readRecord<MCPersistedWorkspace>(WORKSPACE_KEY);
}

export async function saveMCWorkspace(
  workspace: MCPersistedWorkspace,
): Promise<void> {
  await writeRecord(WORKSPACE_KEY, workspace);
}

export async function loadMCResult(): Promise<MCPipelineResult | null> {
  return readRecord<MCPipelineResult>(RESULT_KEY);
}

export async function saveMCResult(result: MCPipelineResult): Promise<void> {
  await writeRecord(RESULT_KEY, result);
}

export async function clearMCWorkspace(): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).clear();
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
  db.close();
}

async function readRecord<T>(key: string): Promise<T | null> {
  const db = await openDatabase();
  const value = await new Promise<T | undefined>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readonly");
    const request = transaction.objectStore(STORE_NAME).get(key);
    request.onsuccess = () => resolve(request.result as T | undefined);
    request.onerror = () => reject(request.error);
  });
  db.close();
  return value ?? null;
}

async function writeRecord(key: string, value: unknown): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put(value, key);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
  db.close();
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
