# Deployment: Docker Hub + Runpod backend, Render frontend

## 1. Publish the backend image

From the repository root, use a Linux image matching the Runpod host:

```bash
docker login
docker buildx build \
  --platform linux/amd64 \
  -t <dockerhub-user>/neural-echo-api:latest \
  --push .
```

The image exposes port `8000`, starts FastAPI with Uvicorn, and includes a
Docker health check at `/health`.

## 2. Start it on Runpod

Create a GPU Pod from `<dockerhub-user>/neural-echo-api:latest` and configure:

- Expose HTTP port `8000`.
- Mount the persistent network volume at `/app/data`. TRIBE/Hugging Face model
  caches, job databases, uploaded references, and generated audio persist here.
- Set `ELEVENLABS_API_KEY` and `ANTHROPIC_API_KEY` as environment variables.
- Set `PORT=8000` if the template overrides the image default.
- Use at least 30 GB of volume storage for model caches and generated jobs.

After the container becomes healthy, verify:

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```

The first boot is intentionally slower because the lifespan hook downloads and
warms TRIBE before `/health` returns. Subsequent boots use `/app/data/cache`.

## 3. Deploy the frontend on Render

`render.yaml` defines the Node service with a reproducible `npm ci` build. Set
`NEXT_PUBLIC_API_URL` to the Runpod proxy URL without a trailing slash, then
trigger a full Render deploy. This value is compiled into the browser bundle;
changing it requires a rebuild, not only a service restart.

The Plotly renderer is client-only and lazy-loaded, so Render never attempts to
access WebGL during SSR. Browser requests go directly to Runpod; the FastAPI
CORS policy permits the Render origin. The 10-frame cortical replay is roughly
8.7 MB as JSON and about 2 MB over the wire through endpoint-level gzip.
