# Deployment guide — GitHub + Streamlit Cloud

## 1. Push to GitHub

Open a terminal (PowerShell) in this folder (`dynamic-pricing-engine`) and run:

```powershell
git init
git add .
git commit -m "Dynamic pricing engine: demand modeling + price optimization"
```

Create a new **public** repo on github.com named `dynamic-pricing-engine` (no README), then:

```powershell
git remote add origin https://github.com/<your-username>/dynamic-pricing-engine.git
git branch -M main
git push -u origin main
```

## 2. Deploy on Streamlit Cloud (free)

1. Go to https://share.streamlit.io and sign in **with GitHub**.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-username>/dynamic-pricing-engine`, branch `main`, main file `app.py`.
4. Click **Deploy**. First build takes ~2 minutes (it installs requirements and trains the model on first load).

You'll get a permanent URL like `https://<your-username>-dynamic-pricing-engine.streamlit.app`.

## 3. Add to LinkedIn

- Paste the live URL into the README (`Live demo` line), commit and push.
- On LinkedIn: Profile → **Add profile section → Featured → Link** → paste the Streamlit URL.
- Also add it under **Projects** with a short description, e.g.:

> *Dynamic Pricing Engine — end-to-end ML system that learns product demand curves with gradient boosting and recommends profit-maximizing prices. Live interactive demo + open-source code.*

Tip: post a short LinkedIn post with a screen recording of you moving the price slider — interactive demos get far more engagement than repo links.
