import { loader } from "@monaco-editor/react";
import type { Monaco } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import { configureMonacoYaml } from "monaco-yaml";
import {
  symbolicDsgeConfigModelPath,
  symbolicDsgeConfigSchema,
  symbolicDsgeSchemaUri,
} from "./configSchema";
import YamlWorker from "./yaml.worker?worker";

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

loader.config({ monaco });

let yamlConfigured = false;

export function configureSymbolicDsgeYaml(editorMonaco: Monaco) {
  if (yamlConfigured) return;

  configureMonacoYaml(editorMonaco, {
    completion: true,
    enableSchemaRequest: false,
    format: { enable: true },
    hover: true,
    validate: true,
    schemas: [
      {
        uri: symbolicDsgeSchemaUri,
        fileMatch: [symbolicDsgeConfigModelPath, "**/*.yaml", "**/*.yml"],
        schema: symbolicDsgeConfigSchema,
      },
    ],
  });
  yamlConfigured = true;
}
