---
tags:
    - doc
---
# ModelParser

```python
class ModelParser(config_path: str | pathlib.Path)
```

`#!python ModelParser` reads a given YAML configuration file (see [configuration guide](../Guides/model_config_guide.md)) and parses the contents into a [`ModelConfig`](../Documentation/ModelConfig.md) object.

__Parameters:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| config_path | `#!python str | pathlib.Path` | Path to the YAML config file.
| config | `ModelConfig` | Parsed config object. |

&nbsp;

__Methods:__

| __Signature__ | __Return Type__ | __Description__ |
|:--------------|:---------------:|----------------:|
| `#!python .from_yaml()` | `#!python ParsedConfig` | Reads the file and populates `#!python ModelParser.parsed`. Runs at `#!python __init__`. |
| `#!python .get()` | `#!python ModelConfig` | Returns the currently parsed model config. |
| `#!python .get_all()` | `#!python ParsedConfig` | Returns the currently parsed model/kalman config pair. |
| `#!python .to_pickle(filepath: str | pathlib.Path)` | `#!python None` | Serializes the current `#!python ModelConfig` and saves to `#!python filepath`.|

???+ note "ParsedConfig"
    `ParsedConfig` is an unpack-able `dataclass` holding `#!python ParsedConfig.model: ModelConfig` and `#!python ParsedConfig.kalman: KalmanConfig`
