import type { JSONSchema } from "monaco-yaml";

const symbolMap: JSONSchema = {
  type: "object",
  additionalProperties: { type: "string" },
};

const numberMap: JSONSchema = {
  type: "object",
  additionalProperties: { type: "number" },
};

const boolMap: JSONSchema = {
  type: "object",
  additionalProperties: { type: "boolean" },
};

const expressionMap: JSONSchema = {
  type: "object",
  additionalProperties: { type: "string" },
};

export const symbolicDsgeConfigSchema: JSONSchema = {
  $id: "https://symbolicdsge.local/schemas/model-config.schema.json",
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "SymbolicDSGE Model Config",
  type: "object",
  required: [
    "variables",
    "parameters",
    "shock_map",
    "observables",
    "equations",
    "calibration",
  ],
  properties: {
    name: {
      type: "string",
      description: "Model name shown in summaries and generated artifacts.",
    },
    variables: {
      description:
        "Ordered model variables. Use a list for plain variables or a mapping for linearization metadata.",
      oneOf: [
        {
          type: "array",
          items: { type: "string" },
          uniqueItems: true,
        },
        {
          type: "object",
          additionalProperties: {
            oneOf: [
              { type: "null" },
              {
                type: "object",
                additionalProperties: false,
                properties: {
                  linearization: {
                    type: "string",
                    enum: ["none", "log", "taylor"],
                    description:
                      "Symbolic linearization method used when compile(linearize=True) is requested.",
                  },
                  steady_state: {
                    oneOf: [{ type: "string" }, { type: "number" }, { type: "null" }],
                    description:
                      "Steady-state expression required for log or Taylor linearization.",
                  },
                },
              },
            ],
          },
        },
      ],
    },
    constrained: {
      ...boolMap,
      description: "Map from variable name to whether the variable has a constraint equation.",
    },
    parameters: {
      type: "array",
      items: { type: "string" },
      uniqueItems: true,
      description: "Declared parameter names. Each must have a calibration value.",
    },
    shock_map: {
      ...symbolMap,
      description: "Map from shock symbols to exogenous state variables.",
    },
    observables: {
      type: "array",
      items: { type: "string" },
      uniqueItems: true,
      description: "Observable names used by measurement equations and Kalman setup.",
    },
    equations: {
      type: "object",
      required: ["model"],
      additionalProperties: false,
      properties: {
        model: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
          description: "Model equations written as SymPy-parseable equalities.",
        },
        constraint: {
          ...expressionMap,
          description: "Map from constrained variable name to relational constraint.",
        },
        observables: {
          ...expressionMap,
          description: "Map from observable name to measurement expression.",
        },
      },
    },
    calibration: {
      type: "object",
      required: ["parameters"],
      additionalProperties: false,
      properties: {
        parameters: {
          ...numberMap,
          description: "Numeric calibration values keyed by declared parameter name.",
        },
        shocks: {
          type: "object",
          additionalProperties: false,
          properties: {
            std: {
              ...symbolMap,
              description: "Map from shock name to standard-deviation parameter name.",
            },
            corr: {
              ...symbolMap,
              description:
                "Map from comma-separated shock pair, e.g. 'e_g, e_z', to correlation parameter name.",
            },
          },
        },
      },
    },
    kalman: {
      type: "object",
      additionalProperties: false,
      properties: {
        y: {
          type: "array",
          items: { type: "string" },
          description: "Measurement ordering. Defaults to observables when omitted.",
        },
        R: {
          type: "object",
          additionalProperties: false,
          properties: {
            std: {
              ...symbolMap,
              description: "Map from observable name to measurement std parameter.",
            },
            corr: {
              ...symbolMap,
              description:
                "Map from comma-separated observable pair to measurement correlation parameter.",
            },
          },
        },
        P0: {
          type: "object",
          additionalProperties: false,
          properties: {
            mode: {
              type: "string",
              enum: ["diag", "eye"],
            },
            scale: { type: "number" },
            diag: {
              ...numberMap,
              description: "Initial covariance diagonal values keyed by state variable.",
            },
          },
        },
        jitter: { type: "number" },
        symmetrize: { type: "boolean" },
      },
    },
  },
  additionalProperties: false,
};

export const symbolicDsgeSchemaUri =
  "https://symbolicdsge.local/schemas/model-config.schema.json";

export const symbolicDsgeConfigModelPath = "file:///symbolicdsge-config.yaml";
