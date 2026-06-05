import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import YamlWorker from "monaco-yaml/yaml.worker?worker";

type MonacoWorkerEnvironment = {
  getWorker: (_moduleId: string, label: string) => Worker;
};

(globalThis as typeof globalThis & { MonacoEnvironment: MonacoWorkerEnvironment })
  .MonacoEnvironment = {
  getWorker(_moduleId: string, label: string) {
    if (label === "yaml") {
      return new YamlWorker();
    }
    return new EditorWorker();
  },
};
