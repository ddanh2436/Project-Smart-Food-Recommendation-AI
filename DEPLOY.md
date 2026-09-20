# VietNomNom — Deployment Runbook

Deploy order matters, because each service needs the previous one's URL:

```
1. MongoDB Atlas   →  MONGO_URI
2. AI (HF Spaces)  →  needs MONGO_URI            →  gives AI_SERVICE_URL
3. Backend (Render)→  needs MONGO_URI + AI url   →  gives BACKEND_URL
4. Frontend(Vercel)→  needs BACKEND_URL          →  gives FRONTEND_URL
5. Go back and fill FRONTEND_URL into Render + Google OAuth
```

Step 5 is the one people forget: the backend cannot know the frontend's URL until
the frontend exists, so CORS and the Google redirect stay broken until you
circle back.

Everything here is free tier. No paid API keys are used anywhere.

---

## 0. Before you start

Create the three `.env` files from their templates and fill in real values:

```bash
cp AI/.env.example       AI/.env
cp Backend/.env.example  Backend/.env
cp Frontend/.env.example Frontend/.env.local
```

Generate fresh secrets (do not reuse the placeholder values):

```bash
node -e "console.log(require('crypto').randomBytes(48).toString('base64url'))"
```

Run it four times, for `JWT_SECRET`, `JWT_REFRESH_SECRET`, `ADMIN_TOKEN`
(Backend) and `ADMIN_TOKEN` (AI). `JWT_SECRET` and `JWT_REFRESH_SECRET` **must
differ** — the backend refuses to start if they match, because an access token
would otherwise be accepted as a refresh token.

> `.env` files are gitignored. `.env.example` is committed. Never swap that.

---

## 1. MongoDB Atlas

1. <https://cloud.mongodb.com> → your cluster → **Connect** → **Drivers**.
2. Copy the connection string and insert the database name before the `?`:
   ```
   mongodb+srv://user:pass@cluster.mongodb.net/VietNomNom?retryWrites=true&w=majority
   ```
3. **Network Access** → **Add IP Address** → `0.0.0.0/0`.

   Render and Hugging Face do not publish fixed egress IPs on free tiers, so an
   allow-list of specific addresses will intermittently fail. If that is too
   open for you, upgrade to a paid tier that offers static IPs or VPC peering.

4. Confirm the `restaurants` and `reviews` collections are present and populated.

---

## 2. AI service → Hugging Face Spaces

The Space runs the FastAPI app from a Dockerfile.

### Create it

1. <https://huggingface.co/new-space>
2. **SDK: Docker** → *Blank*. Hardware: **CPU basic (free)**.
3. Push the code:

   ```bash
   cd AI
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```

   If the Space already exists with history, use `git push space main --force`
   only when you are sure you want to replace it.

### Configure it

**Settings → Variables and secrets**:

| Name | Type | Value |
|---|---|---|
| `MONGO_URI` | **Secret** | the Atlas string from step 1 |
| `ADMIN_TOKEN` | **Secret** | a generated random string |
| `DB_NAME` | Variable | `VietNomNom` |
| `PORT` | Variable | `7860` |
| `CORS_ORIGINS` | Variable | `*` (only the backend calls it) |
| `ENABLE_SEMANTIC_SEARCH` | Variable | `true` |

### Verify

First build takes 10-20 minutes (torch is large). First *boot* then downloads
the sentiment and embedding models, another 2-5 minutes.

```bash
curl https://<user>-<space-name>.hf.space/health
```

Expect `"status": "ok"` and a non-zero `data.restaurants`. Check each model:

```json
{ "models": { "sentiment": {"ready": true},
              "yolo": {"ready": true},
              "semantic_search": {"ready": true} } }
```

**Your `AI_SERVICE_URL` is `https://<user>-<space-name>.hf.space`** — note the
`.hf.space` host, not the `huggingface.co/spaces/...` page URL.

### If the Space runs out of memory

Set `ENABLE_SEMANTIC_SEARCH=false`. That drops the ~500MB embedding model;
search falls back to TF-IDF and everything else keeps working.

### Free-tier sleep

A free Space sleeps after ~48h idle and cold-starts on the next request, which
can take ~50s. The backend's HTTP timeouts allow for this and every AI call
degrades gracefully, so the site stays usable while the Space wakes up.

---

## 3. Backend → Render

### Create it

1. <https://dashboard.render.com> → **New** → **Web Service** → connect the
   backend repo. (`Backend/render.yaml` also works via **New → Blueprint**.)
2. Settings:
   - Build command: `npm ci && npm run build`
   - Start command: `npm run start:prod`
   - Health check path: `/health`
   - Region: **Singapore** (closest to Vietnamese users)

### Environment variables

| Name | Value |
|---|---|
| `MONGODB_URI` | the Atlas string (note: `MONGODB_URI`, not `MONGO_URI`) |
| `JWT_SECRET` | generated |
| `JWT_REFRESH_SECRET` | generated, **different** from the above |
| `ADMIN_TOKEN` | generated |
| `AI_SERVICE_URL` | `https://<user>-<space>.hf.space` |
| `BACKEND_URL` | `https://<service>.onrender.com` |
| `FRONTEND_URL` | fill in after step 4 |
| `CORS_ORIGINS` | fill in after step 4 |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | from step 5 |
| `NODE_ENV` | `production` |

The service **exits on boot** if `MONGODB_URI`, `JWT_SECRET` or
`JWT_REFRESH_SECRET` is missing — deliberately, so a misconfigured deploy fails
visibly instead of serving 500s that look like a healthy service.

### Verify

```bash
curl https://<service>.onrender.com/health
```

Expect `"database": { "state": "connected" }`.

---

## 4. Frontend → Vercel

1. <https://vercel.com/new> → import the frontend repo.
2. **Root directory**: leave as the repo root (the repo *is* the frontend).
3. Environment variables:

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<service>.onrender.com` |
   | `NEXT_PUBLIC_SITE_URL` | `https://<project>.vercel.app` |

   No trailing slashes. Anything prefixed `NEXT_PUBLIC_` is embedded in the
   browser bundle, so never put a secret there.

4. Deploy.

---

## 5. Close the loop ← do not skip

### 5a. Tell the backend about the frontend

In Render, set both and redeploy:

```
FRONTEND_URL = https://<project>.vercel.app
CORS_ORIGINS = https://<project>.vercel.app
```

Without this, every browser request fails CORS. Vercel *preview* deployments
(`*.vercel.app` per commit) are matched by a pattern, so you only need the
production URL here.

### 5b. Google OAuth

<https://console.cloud.google.com> → **APIs & Services → Credentials** → your
OAuth 2.0 Client ID:

- **Authorized redirect URIs**: `https://<service>.onrender.com/auth/google/callback`
- **Authorized JavaScript origins**: `https://<project>.vercel.app`

The redirect URI must match byte for byte — no trailing slash, `https` not
`http`. Then put the client ID and secret into Render.

### 5c. Backfill review sentiment (once)

Labels older reviews so the AI analytics on restaurant pages have data:

```bash
curl -X POST https://<service>.onrender.com/reviews/migrate-sentiment \
     -H "x-admin-token: <your ADMIN_TOKEN>"
```

This endpoint requires the admin token — it used to be public, which let anyone
trigger a full-collection scan plus an AI call per row.

---

## 6. Smoke test the deployment

| Check | Expected |
|---|---|
| Home page loads with restaurant cards | data flows Vercel → Render → Atlas |
| Search "bún bò huế quận 1" | AI intent parsing + ranking works |
| Search "xyzabc" | returns *no* results, not the whole database |
| Chatbot: "phở ở Hà Nội" then "rẻ hơn đi" | second reply keeps pho + Hanoi, re-sorts by price |
| Upload a dish photo | YOLO recognises it and lists matching places |
| A restaurant page with reviews | shows the aspect breakdown card |
| Register → refresh the page after 20 min | still signed in (token refresh works) |
| Log out, then reuse the old refresh token | rejected |

---

## Common failures

| Symptom | Cause |
|---|---|
| CORS error in the browser console | `CORS_ORIGINS` / `FRONTEND_URL` not set on Render (step 5a), or has a trailing slash |
| Search returns nothing at all | AI Space asleep or `AI_SERVICE_URL` wrong — check `/health` on both services |
| `AI_SERVICE_URL` "works" but search is empty | you used the `huggingface.co/spaces/...` page URL instead of `<user>-<space>.hf.space` |
| Google login → `redirect_uri_mismatch` | the URI in Google Console differs from `BACKEND_URL/auth/google/callback` |
| Backend exits immediately on deploy | a required env var is missing; the Render log names which one |
| Logged out every 15 minutes | `JWT_REFRESH_SECRET` missing or equal to `JWT_SECRET` |
| AI Space build fails on torch | free-tier disk limit; the Dockerfile already installs the CPU-only wheel — confirm the `--extra-index-url` line survived in `requirements.txt` |
| First request after idle takes ~50s | normal free-tier cold start (Render *and* Hugging Face both sleep) |

---

## Local development

Three terminals:

```bash
# 1. AI service  → http://127.0.0.1:5000/docs
cd AI && pip install -r requirements.txt && python api.py

# 2. Backend     → http://localhost:3001/health
cd Backend && npm install && npm run start:dev

# 3. Frontend    → http://localhost:3000
cd Frontend && npm install && npm run dev
```

Tests:

```bash
cd AI       && python test_api.py   # offline, no Mongo needed
cd Backend  && npm test             # unit tests
cd Backend  && npm run typecheck
cd Frontend && npx tsc --noEmit && npm run build
```
