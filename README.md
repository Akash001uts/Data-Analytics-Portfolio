# Data Analytics Portfolio

[![CI](https://github.com/Akash001uts/Data-Analytics-Portfolio/actions/workflows/ci.yml/badge.svg?branch=rebuild)](https://github.com/Akash001uts/Data-Analytics-Portfolio/actions/workflows/ci.yml)

I'm a uni student studying data analytics, and this is where I keep my data projects. The first version of this
repo had three projects I made while I was learning: customer segmentation, sentiment analysis and an LSTM stock
forecast. Looking back at them, they mostly followed well-known tutorials, they didn't report proper metrics, and a
couple of them had real bugs. So I'm rebuilding it around projects where I ask my own question, use real open data,
and check my work properly. The old version is still at the [`v1-coursework`](https://github.com/Akash001uts/Data-Analytics-Portfolio/tree/v1-coursework)
tag if you want to see where I started.

| Project | The question | Status |
| --- | --- | --- |
| [Avoidable hospital admissions](projects/avoidable-hospitalisations) | Which parts of Australia have more potentially preventable hospital admissions than you'd expect from their social and access profile? | Data sourced and checked, analysis in progress |
| [Sentiment analysis, redone](projects/sentiment-analysis) | When you measure it properly, how much better is a transformer than simple baselines at reading food reviews? | Planned |

## What I've learnt so far

I keep a running log in [LEARNINGS.md](LEARNINGS.md). The short version:

- **Check what the data actually counts before you pick it.** The most detailed hospital data I found only covers
  public hospitals, and that gap lines up with private health insurance. If I'd used it, my main map would have partly
  been a map of who has private cover.
- **Pin your data.** One of my sources published a new release while I was working on it. Because every file is
  checked against a fingerprint (a SHA256 hash), the download stopped and told me, instead of quietly changing my results.
- **Write the "don't cheat" rules as tests.** The easiest way to get a great-looking model is to accidentally feed it
  the answer. Here, the list of allowed inputs is in code, and tests fail if anything hospital-related sneaks in.
- **Look at the rows, not just the headers.** The spreadsheets had fake "areas" mixed in with the real ones, which I
  only found when I made a test count the rows.

## How the repo is laid out

```
projects/
  avoidable-hospitalisations/   the question, approach, findings so far, and data.md (every source and check)
  sentiment-analysis/           the plan for the rebuilt sentiment project
src/dap/                        the shared Python code, so tests and notebooks can import it
  common/                       file paths, seeds, and the data manifest checker
  health/                       downloading the data, the feature allowlist, reading the PHIDU workbook
tests/                          pytest tests, plus tiny made-up data in tests/fixtures
data/manifest.yaml              where every data file comes from, with its size and SHA256 (the data itself isn't in git)
LEARNINGS.md                    what I've learnt and got wrong along the way
```

## Reproduce it yourself

Everything here runs on a normal laptop, with no GPU and no accounts to sign up for. All the data is public.

### Option 1: run it in your browser

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Akash001uts/Data-Analytics-Portfolio?ref=rebuild)

This opens the repo in a GitHub Codespace (free for personal accounts, within GitHub's monthly limit). It installs
everything for you. When it's ready, run the commands from step 3 below in its terminal.

### Option 2: run it on your own machine

1. **Install uv**, which handles Python and all the packages for you. On Windows (PowerShell):

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   On macOS or Linux:

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   (If you'd rather use pip, `pip install uv` works too. Just make sure the folder it installs into is on your PATH.)

2. **Get the code.** While I'm rebuilding, the new version lives on the `rebuild` branch:

   ```
   git clone --branch rebuild https://github.com/Akash001uts/Data-Analytics-Portfolio.git
   cd Data-Analytics-Portfolio
   uv sync
   ```

   `uv sync` downloads Python 3.12 if you don't have it and installs the exact package versions in `uv.lock`.

3. **Run the tests.**

   ```
   uv run pytest
   ```

   These run on small made-up data in `tests/fixtures`, so they take a couple of seconds. They check that the data
   manifest is complete, that downloads get verified, and that the feature allowlist blocks anything that would leak
   the answer into the model. A few tests say "skipped" because they need the real data, which comes next.

4. **Download the real data.**

   ```
   uv run dap health fetch
   ```

   This downloads about 210 MB from the AIHW, ABS and PHIDU websites into `data/raw/` and checks every file against
   `data/manifest.yaml`. Run `uv run pytest` again afterwards and the skipped tests will run too. They check every
   feature I use against the real spreadsheets.

5. **What isn't built yet.** `dap health build`, `train` and `report` will clean the data, fit the models and make the
   maps. For now they just tell you which phase they're planned for. I'll update this section as they're built.

### If something goes wrong

- **`upstream changed: ...`** means a publisher has updated a file since I last checked it, so it no longer matches
  its fingerprint. Nothing is wrong with your setup. The file that failed gets kept as `*.mismatch` in `data/raw/` so
  you can look at it. Feel free to [open an issue](https://github.com/Akash001uts/Data-Analytics-Portfolio/issues)
  and I'll re-check the source.
- **`uv` not found** after a pip install usually means pip's scripts folder isn't on your PATH. The official installer
  in step 1 avoids this.
- **Downloads are slow or blocked:** the files come from government websites, so a work or uni network that blocks
  large downloads can get in the way. Try another network, or download a single source with
  `uv run dap health fetch --only aihw_pph_sa3`.

## Data and credits

The data comes from the Australian Institute of Health and Welfare (AIHW), the Australian Bureau of Statistics (ABS)
and the Public Health Information Development Unit (PHIDU) at Torrens University Australia. Every source, its edition
and its licence is listed in [the project's data notes](projects/avoidable-hospitalisations/data.md).

Based on Public Health Information Development Unit (PHIDU), Torrens University Australia material from: Social Health
Atlas of Australia: Population Health Areas (online) 2026. AIHW and ABS material is used under CC BY 4.0.

## Licence

The code is MIT licensed. Anything derived from PHIDU data (tables, figures, maps and results) is shared under
CC BY-NC-SA 3.0 AU, as PHIDU's licence requires, so it's for non-commercial use.

This is a personal student project and isn't affiliated with or endorsed by any of the data publishers.
