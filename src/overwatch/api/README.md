# api: the web interface's backend

```bash
uv run python -m overwatch.api            # then open http://127.0.0.1:8000
uv run python -m overwatch.api --gpu-layers 0   # keep the judge off the GPU
```

| endpoint | what it does |
|---|---|
| `GET /` | the page (`overwatch/web/index.html`; design notes in `web/DESIGN.md` beside it) |
| `GET /api/status` | whether a scorer and a judge model are available, and if a job is running |
| `POST /api/analyses?name=&judge=` | the demo as the raw request body; starts an analysis |
| `GET /api/analyses/{id}` | a job's stage and a sentence describing it |
| `GET /api/reports` | saved reports, newest first |
| `GET /api/reports/{id}` | one report: `pipeline.MatchReport` as JSON |

How it behaves:

- **One analysis at a time**, on a background thread (`jobs.Runner`). The judge
  holds a language model in memory; two analyses sharing it would each be slower.
  Models load on first use and stay loaded.
- **Uploads stream to disk** under their content hash in `data/uploads/`. The
  browser's filename is only displayed. Anything not starting with CS2's
  `PBDEMS2` header is refused before analysis.
- **Reports** are the pipeline's JSON in `data/reports/`, the same files
  `python -m overwatch.pipeline` writes, so both show up in the page's history.
- **Local only**: bound to 127.0.0.1, no accounts.
