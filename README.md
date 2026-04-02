# Repo Historian

A LangGraph pipeline that reads a GitHub repository's commit history and produces a **cited narrative history** — a prose document telling the story of how the project evolved, with inline commit-range citations linking back to GitHub diffs.

Supports single-repo and multi-repo modes, and works with Claude, GPT, and Gemini models.

## Pipeline

<img width="1078" height="756" alt="Screenshot 2026-04-02 at 11 21 07 AM" src="https://github.com/user-attachments/assets/ca59dc74-e740-4e19-a3d0-21d65d7479df" />


### Single repo

```
fetch metadata → fetch commits → triage (LLM picks inflection points)
→ analyze diffs (fan-out, parallel) → write narrative
```

### Multi-repo

```
fan out per-repo pipelines (fetch → triage → analyze, in parallel)
→ collect & merge analyses → write cross-repo narrative
```

## Setup

```bash
uv sync
cp .env.example .env
# Fill in GITHUB_TOKEN and the API key for your chosen provider
```

## Usage

```bash
# Single repo
uv run python -m repo_historian https://github.com/owner/repo

# Multiple repos — produces a unified cross-repo narrative
uv run python -m repo_historian --repos https://github.com/org/repo1 https://github.com/org/repo2

# Custom output filename
uv run python -m repo_historian https://github.com/owner/repo --name my-project

# Custom narrative style
uv run python -m repo_historian https://github.com/owner/repo --style "a Ken Burns documentary"
```

## Output

Written to `output/`:

| File | Contents |
|------|----------|
| `{slug}_narrative.md` | Prose history with inline commit-range citations |
| `{slug}_raw.json` | Structured data: repo metadata and diff analyses |

## Configuration

Edit `repo_historian/config.py` to change:

- **`MODEL_NAME`** — LLM used for triage and diff analysis
- **`NARRATIVE_MODEL_NAME`** — LLM used for narrative generation (can differ from the data model)
- **Triage parameters** — window size, margin, inflection point bounds
- **Diff caps** — max files per diff, max patch size
- **Narrative limits** — min/max word counts
- **Temperature and token limits** — separately configurable for data and narrative models

Provider is auto-detected from the model name. Set the corresponding API key in your `.env`:

| Provider | Model prefixes | Env var |
|----------|---------------|---------|
| Anthropic | `claude-*` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-*`, `o-*` | `OPENAI_API_KEY` |
| Google | `gemini-*` | `GOOGLE_API_KEY` |

LangSmith tracing is supported — see `.env.example` for the optional config.

## License

[MIT](LICENSE)
