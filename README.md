# Data Analytics Portfolio

I'm rebuilding this repository. It will hold two projects:

- **Avoidable hospital admissions across Australia** (in progress): which areas have more potentially
  preventable hospitalisations than their social, demographic and access profile predicts. It uses open
  AIHW, PHIDU and ABS data, with spatial cross-validation and a map of residuals.
- **Sentiment analysis, re-evaluated** (planned): rule-based, linear and transformer models on Amazon
  Fine Food Reviews, measured properly with macro-F1, baselines and error analysis.

Data sources, editions, licences and the feature allowlist are documented in [docs/data.md](docs/data.md).

## Earlier coursework (v1)

My earlier coursework (customer segmentation, the first sentiment analysis and an LSTM stock forecast)
is preserved at the [`v1-coursework`](https://github.com/Akash001uts/Data-Analytics-Portfolio/tree/v1-coursework)
tag.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
uv run dap health fetch   # downloads the raw data and checks it against data/manifest.yaml
```

## Licence

Code is MIT licensed. Outputs derived from PHIDU data are CC BY-NC-SA 3.0 AU; see
[docs/data.md](docs/data.md#licences-and-attribution).
