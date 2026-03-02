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
| config_path | `#!python str \| pathlib.Path` | Path to the YAML config file. |
| parsed | `#!python ParsedConfig` | Parsed model/kalman config pair populated at initialization. |

&nbsp;

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python .from_yaml()` | `#!python ParsedConfig` | Reads the file and populates `#!python ModelParser.parsed`. Runs at `#!python __init__`. |
| `#!python .get()` | `#!python ModelConfig` | Returns the currently parsed model config. |
| `#!python .get_all()` | `#!python ParsedConfig` | Returns the currently parsed model/kalman config pair. |

???+ note "ParsedConfig"
    `ParsedConfig` is an unpack-able `dataclass` holding `#!python ParsedConfig.model: ModelConfig` and `#!python ParsedConfig.kalman: KalmanConfig`

???+ warning "Calibration Completeness"
    Parsing enforces that every declared parameter has a calibration value, and every parameter referenced in shock/Kalman sections is declared and calibrated.
