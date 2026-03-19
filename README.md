# Repo Historian

A LangGraph pipeline that reads a GitHub repo's full commit history and produces a **cited narrative history** — a prose document telling the story of how the project evolved, with `[[sha]](url)` inline citations.

## Pipeline

```
fetch metadata → fetch commits → triage (LLM picks significant commits)
→ analyze commits (fan-out, parallel) → cluster into 3-7 eras
→ build outline → expand to narrative
```

## Usage

```bash
# Required env vars (in .env or shell)
export GITHUB_TOKEN=...
export ANTHROPIC_API_KEY=...  # or OPENAI_API_KEY / GOOGLE_API_KEY

uv run python -m repo_historian https://github.com/owner/repo [--batch-size 80]
```

## Output

Written to `output/`:

| File | Contents |
|------|----------|
| `owner_repo_outline.md` | Structured outline: eras, commits, diffs |
| `owner_repo_narrative.md` | Prose history with inline commit citations |
| `owner_repo_raw.json` | All data (metadata, analyses, eras) |

## Configuration

Edit `repo_historian/config.py` to change the LLM model (`MODEL_NAME`), triage batch size, diff size caps, era count bounds, or narrative word limits.

Supports Claude, GPT, and Gemini models — provider is auto-detected from the model name.
