# Deploying to Hugging Face Spaces (free CPU tier)

The Space is its own git repo on huggingface.co. We build it fresh with Git LFS
so the model and graph upload correctly, and export the app from this project
with `git archive` (which ships only tracked files - no venv, no caches, and
crucially no `.streamlit/secrets.toml`, since it is gitignored).

Replace `<user>` with your Hugging Face username throughout.

## 1. Create the Space (web, one time)
On https://huggingface.co → New → Space:
- Owner: you · Name: `emergency-routing`
- SDK: **Docker** · Template: blank
- Hardware: **CPU basic (free)** · Visibility: Public (or Private)

## 2. Build the Space repo locally
Run from anywhere (adjust the project path if different):

    git clone https://huggingface.co/spaces/<user>/emergency-routing hf-space
    cd hf-space
    git lfs install

    # LFS rules and the HF README must exist before adding the big files
    cp /e/traffic-final/deploy/hf/.gitattributes .
    cp /e/traffic-final/deploy/hf/README.md .

    # Export only tracked project files into here
    git -C /e/traffic-final archive HEAD | tar -x

    # The project readme is README.MD; HF reads README.md (already copied above)
    rm -f README.MD

    git add -A
    git commit -m "Deploy Emergency Routing FastAPI app"
    git push

The first push uploads the LFS files (~55 MB); HF then builds the Docker image
and starts it. Build + first boot takes a few minutes. Your app will be at:
`https://<user>-emergency-routing.hf.space`

## 3. (Optional) Enable the vision features
In the Space → Settings → **Variables and secrets**, add:
- Secret `ANTHROPIC_API_KEY` = your key
- Variable `LLM_PROVIDER` = `anthropic`

The app reads these from the environment; no key is ever committed. Restart the
Space (Settings → Factory reboot) after adding them.

## 4. Updating later
From `hf-space`, re-run the `git archive` + `rm README.MD` + commit + push steps,
or just push the specific files you changed. HF rebuilds automatically on push.

## Notes / gotchas
- Free Spaces sleep after inactivity and cold-start on the next visit (~10-30 s
  while the model and graph load). That is expected on the free tier.
- If the build fails on a cache write as the non-root user, add `ENV HOME=/tmp`
  to the Dockerfile before the CMD and push again.
- The Space is a separate repo from GitHub; keeping them in sync is manual
  (re-run step 4), which is fine for a demo.
