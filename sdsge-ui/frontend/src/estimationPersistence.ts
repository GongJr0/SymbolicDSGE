import type {
  EstimationMethod,
  EstimationParameterSpec,
  EstimationRunResult,
  Role,
} from "./types";

const DB_NAME = "symbolicdsge-estimation-ui";
const DB_VERSION = 1;
const STORE_NAME = "estimation-workspace";

export interface EstimationPersistedWorkspace {
  version: 1;
  role: Role;
  modelKey: string;
  method: EstimationMethod;
  parameters: EstimationParameterSpec[];
  selected: string | null;
  observables: string;
  dataVectors: Record<string, string>;
  maxIter: number;
  nDraws: number;
  burnIn: number;
  thin: number;
  seed: number;
  adapt: boolean;
  proposalScale: number;
  posteriorPoint: string;
  result: EstimationRunResult | null;
  modeFolded: boolean;
}

export async function loadEstimationWorkspace(
  role: Role,
): Promise<EstimationPersistedWorkspace | null> {
  return readRecord<EstimationPersistedWorkspace>(role);
}

export async function saveEstimationWorkspace(
  workspace: EstimationPersistedWorkspace,
): Promise<void> {
  await writeRecord(workspace.role, workspace);
}

export async function clearEstimationWorkspace(role: Role): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).delete(role);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
  db.close();
}

async function readRecord<T>(key: IDBValidKey): Promise<T | null> {
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

async function writeRecord(key: IDBValidKey, value: unknown): Promise<void> {
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
