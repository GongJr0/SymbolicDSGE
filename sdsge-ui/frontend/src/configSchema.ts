import type { JSONSchema } from "monaco-yaml";

const symbolMap: JSONSchema = {
  type: "object",
  additionalProperties: { type: "string" },
};

const expressionMap: JSONSchema = {
  type: "object",
  additionalProperties: { type: "string" },
};

/** A calibration value: a number, or an expression over other parameters. */
const numberOrExpressionMap: JSONSchema = {
  type: "object",
  additionalProperties: {
    oneOf: [{ type: "number" }, { type: "string" }],
  },
};

/**
 * One `std`/`corr` covariance block. `calibration.shocks` and
 * `calibration.measurement_noise` share the shape and differ only in what
 * their keys name.
 */
const covarianceSpec = (member: string, pairExample: string): JSONSchema => ({
  type: "object",
  additionalProperties: false,
  properties: {
    std: {
      ...symbolMap,
      description: `Map from ${member} name to its standard-deviation parameter name. Every declared ${member} must appear.`,
    },
    corr: {
      ...symbolMap,
      description: `Map from a comma-separated ${member} pair, e.g. '${pairExample}', to its correlation parameter name. An omitted pair is uncorrelated.`,
    },
  },
});

export const symbolicDsgeConfigSchema: JSONSchema = {
  $id: "https://symbolicdsge.local/schemas/model-config.schema.json",
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "SymbolicDSGE Model Config",
  type: "object",
  required: ["variables", "shocks", "equations", "calibration"],
  additionalProperties: false,
  properties: {
    name: {
      type: "string",
      description: "Model name shown in summaries and generated artifacts.",
    },
    variables: {
      type: "array",
      items: { type: "string" },
      uniqueItems: true,
      description:
        "Ordered model variables, without time indices. Leads and lags beyond the three-date system become auxiliary variables at compile time.",
    },
    ss_seed: {
      ...numberOrExpressionMap,
      description:
        "Newton seed for the steady-state solve, keyed by declared variable. An omitted variable is seeded at zero.",
    },
    shocks: {
      type: "array",
      items: { type: "string" },
      uniqueItems: true,
      description:
        "Innovation symbols, in declaration order, which is the shock column order.",
    },
    equations: {
      type: "object",
      required: ["model"],
      additionalProperties: false,
      properties: {
        model: {
          ...expressionMap,
          minProperties: 1,
          description: "Map from equation name to a SymPy-parseable equality.",
        },
        constraints: {
          type: "object",
          additionalProperties: {
            type: "object",
            additionalProperties: false,
            required: ["bind"],
            properties: {
              bind: {
                type: "string",
                description:
                  "Entry condition, evaluated on the reference path. Relational or boolean combination.",
              },
              relax: {
                type: "string",
                description:
                  "Exit condition. Omitted, it is derived as the negation of `bind`.",
              },
            },
          },
          description: "Map from constraint name to its entry and exit conditions.",
        },
        regimes: {
          type: "object",
          additionalProperties: {
            ...expressionMap,
            minProperties: 1,
          },
          description:
            "Map from comma-joined binding constraint names, e.g. 'elb, irr', to the model equations that regime replaces.",
        },
        observables: {
          ...expressionMap,
          description:
            "Map from observable name to its measurement expression, in current-dated variables.",
        },
      },
    },
    calibration: {
      type: "object",
      required: ["parameters"],
      additionalProperties: false,
      properties: {
        parameters: {
          ...numberOrExpressionMap,
          description:
            "Parameter values keyed by name. A name is a parameter if and only if it is calibrated here. An expression over other parameters is a derived entry: it is inlined into the equations and is not itself estimable.",
        },
        shocks: covarianceSpec("shock", "e_g, e_z"),
        measurement_noise: covarianceSpec("observable", "Infl, Rate"),
      },
    },
  },
};

export const symbolicDsgeSchemaUri =
  "https://symbolicdsge.local/schemas/model-config.schema.json";

export const symbolicDsgeConfigModelPath = "file:///symbolicdsge-config.yaml";
