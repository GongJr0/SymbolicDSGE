---
tags:
    - doc
---
# ModelParser

```python
class ModelParser(config_path: str | pathlib.Path)
```

`#!python ModelParser` reads a given YAML configuration file (see [configuration guide](../guides/model_config_guide.md)) and parses the contents into a [`ModelConfig`](../documentation/ModelConfig.md) object.

__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| config_path | `#!python str | pathlib.Path` | Path to the YAML config file. |
| raw_data | `#!python dict[str, Any]` | The authored YAML mapping, retained for canonical re-emission. |
| parsed | `#!python ParsedConfig` | Parsed model/kalman config pair populated at initialization. |

&nbsp;

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python ModelParser.from_string(text)` | `#!python ModelParser` | Classmethod. Builds a parser from YAML text instead of a path (routed through the same parse pipeline). |
| `#!python .from_yaml()` | `#!python tuple[dict, ParsedConfig]` | Reads the file and populates `#!python ModelParser.raw_data` / `#!python .parsed`. Runs at `#!python __init__`. |
| `#!python .get()` | `#!python ModelConfig` | Returns the currently parsed model config. |
| `#!python .get_all()` | `#!python ParsedConfig` | Returns the currently parsed model/kalman config pair. |
| `#!python .to_yaml(config=None, *, digits=3)` | `#!python str` | Emits the authored configuration as canonical YAML text. When `config` is supplied, its calibration is baked in (rounded to `digits`). |
| `#!python .update_calibration_parameters(new_config, digits=3, output_path=None)` | `#!python io.StringIO` | Convenience wrapper over `to_yaml` that bakes `new_config`'s calibration and optionally writes the result to `output_path`. |

???+ note "ParsedConfig"
    `ParsedConfig` is an unpack-able `dataclass` holding `#!python ParsedConfig.model: ModelConfig` and `#!python ParsedConfig.kalman: KalmanConfig`

???+ info "Serialization Round Trip"
    `to_yaml` re-dumps `raw_data`, so the `(t)`/`(t+1)` equation grammar and any `kalman:` block are preserved verbatim; `ModelParser.from_string(...)` re-parses that text back into an equivalent `ParsedConfig`. The Kalman configuration is not serialized as an object. It rides inside the emitted YAML, and the parser re-derives the model dependent measurement noise machinery on load. This pairing is what the `.sdsge` bundle uses to carry a model's configuration as text.

???+ warning "Calibration Completeness"
    Parsing enforces that every declared parameter has a calibration value, and every parameter referenced in shock/Kalman sections is declared and calibrated.
