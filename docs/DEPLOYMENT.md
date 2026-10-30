# Deploying the dashboard (Streamlit Community Cloud)

The hosted dashboard is a read-only view of saved results. It computes nothing when someone opens it, needs no secrets, and does not use PyTorch.

## What the repository provides

| Item | Where |
|---|---|
| Entry point | `dashboard/app.py` |
| Branch | `main` |
| Dependencies | `requirements.txt` (Streamlit, Plotly, pandas, numpy; deliberately small) |
| Streamlit settings | `.streamlit/config.toml` (theme, no usage statistics, no error details shown to visitors) |
| Results shown | `published/` (committed snapshot; see `published/manifest.json`) |
| Disclaimer | Shown under every page and in the README |

On a fresh clone there is no `outputs/` or `data/` folder, so the dashboard automatically reads `published/`. Raw Binance candles are not published (the Binance data terms restrict redistribution), so the Strategies page shows precomputed paper-account runs instead of the interactive simulator.

## Deploy (about 5 minutes)

You do these steps yourself because they need your GitHub and Streamlit logins.

1. Open <https://share.streamlit.io> and choose **Continue with GitHub**. Sign in as the owner of the repository and authorize Streamlit when GitHub asks. If the repository is not offered, use "Configure GitHub permissions" (or GitHub → Settings → Applications → Streamlit) and grant access to `kronosQuant`.
2. Click **Create app** and choose **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `akv-creates/kronosQuant`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **App URL:** pick a short name, for example `kronosquant`, giving `https://kronosquant.streamlit.app` (the name must be free).
4. Open **Advanced settings** and set **Python version** to **3.12**. Leave the secrets box empty: the app needs none.
5. Click **Deploy**. The first build installs the packages and takes a few minutes.

## Updating after a new experiment

```bash
.venv/bin/python -I scripts/publish_results.py   # rebuilds published/ from your local outputs/
.venv/bin/python -I -m pytest                    # includes the published-snapshot checks
git add published && git commit -m "Update published results" && git push
```

Streamlit Community Cloud redeploys automatically when `main` changes.

## Troubleshooting

- **Build fails while installing:** check that Python 3.12 is selected and that `requirements.txt` has not gained research packages such as PyTorch (they belong in `requirements-research.txt`).
- **An empty or "missing file" page:** run `scripts/publish_results.py` and commit `published/`.
- **Free apps go to sleep** after a period without visitors; the first visit afterwards shows a "wake up" button and takes a few seconds.
- **To test the hosted behaviour locally:** `KRONOSQUANT_PUBLISHED_ONLY=1 .venv/bin/python -I -m streamlit run dashboard/app.py`.
