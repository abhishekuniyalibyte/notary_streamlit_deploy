# NOTARIA_PROJECT — Run Guide

This guide shows how to pull the latest code and run the app using Docker (recommended).

## 1) Prerequisites

### A) Install Git
- Windows: install “Git for Windows”
- macOS: install Xcode Command Line Tools or Git
- Ubuntu/Debian:
  - `sudo apt update && sudo apt install -y git`

Verify:
- `git --version`

### B) Install Python (for local runs)

If you plan to run **without Docker**, install **Python 3.10+**.

Verify:
- `python --version` (Windows) or `python3 --version` (macOS/Linux)

### C) Install Docker

Pick ONE:

- **Windows/macOS (recommended):** install **Docker Desktop** and ensure it’s running.
- **Ubuntu/Linux:** install **Docker Engine**.

Verify:
- `docker --version`
- `docker ps`

If `docker ps` fails with a permissions error on Linux:
- Quick fix (works immediately): prefix commands with `sudo`
- Proper fix (no sudo):
  - `sudo groupadd docker || true`
  - `sudo usermod -aG docker $USER`
  - log out/in (or reboot)
  - `docker ps`

### D) (Optional) Install Docker Compose
Most modern Docker installs include Compose as:
- `docker compose version`

## Windows notes (recommended setup)

If you want to test on Windows, the easiest path is **Docker Desktop (WSL2 backend)**.

- Install **Docker Desktop** and enable **Use the WSL 2 based engine**.
- Keep the repo in a normal local folder (example: `C:\Projects\NOTARIA_PROJECT`) to avoid volume-mount issues.
- If you use `docker compose` (bind mounts), ensure Docker Desktop has access to the drive where the repo lives.

## macOS notes (recommended setup)

On macOS, the easiest path is **Docker Desktop**.

- Install **Docker Desktop for Mac** and ensure it’s running.
- Keep the repo in your user folder (example: `~/Projects/NOTARIA_PROJECT`) to avoid permission issues.

## 2) Get the code (clone / pull)

### First time
```bash
# Replace with your repo URL (or skip if you already have the folder)
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

### Update later
```bash
cd <YOUR_REPO_FOLDER>
git pull
```

## 3) Configure environment (.env)

Create a file named `.env` in the repo root (same folder as `Dockerfile`).

Recommended (macOS/Linux):
```bash
cp .env.example .env
```

Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

Then set:
```bash
GROQ_API_KEY=your_key_here
```

## 4) Run with Docker (recommended)

### Build
```bash
docker build -t notaria:local .
```

### Run
```bash
docker run --rm -p 8501:8501 --env-file .env notaria:local
```

Open in browser:
- `http://localhost:8501`

Stop:
- press `Ctrl+C` in the terminal running the container

## 5) Run with Docker Compose (if you prefer compose)

From the repo root:
```bash
docker compose build
docker compose up
```

Open in browser:
- `http://localhost:8501`

Stop:
```bash
docker compose down
```

## 6) Quick sanity test in the UI

1. Open `http://localhost:8501`
2. Upload 1 document
3. Click **Analyze / check now**
4. Confirm you see the **Document analysis** panel with:
   - `Uploaded files status`
   - `Case completeness status`
   - Missing required documents (if any)

## 7) Common issues

### “permission denied while trying to connect to the docker API at unix:///var/run/docker.sock”
- Use `sudo docker ...` OR add your user to the `docker` group (see Prerequisites → Docker).

### Port already in use
If `8501` is busy, run on a different host port (example `8503`):
```bash
docker run --rm -p 8503:8501 --env-file .env notaria:local
```
Then open:
- `http://localhost:8503`

## 8) Run locally on Windows (no Docker)

This is useful if you want to debug faster, but you may need extra system tools for OCR / scanned PDFs.

### A) Python + virtualenv

PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run chatbot.py
```

If activation fails due to execution policy, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

CMD:
```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run chatbot.py
```

Open in browser:
- `http://localhost:8501`

### B) Optional: OCR + scanned PDF support on Windows

If you need OCR / scanned PDF extraction, install and add to PATH:
- **Tesseract OCR** (plus Spanish language data if needed)
- **Poppler** (for `pdf2image`)
- **LibreOffice** (for legacy `.doc` extraction via `soffice`)

If OCR tools are not installed, the app can still run, but extraction for scanned PDFs / legacy docs may be limited.

## 9) Run locally on macOS (no Docker)

### A) Python + virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run chatbot.py
```

Open in browser:
- `http://localhost:8501`

### B) Optional: OCR + scanned PDF support on macOS

If you need OCR / scanned PDF extraction, install tools via Homebrew:
```bash
brew install tesseract poppler
brew install --cask libreoffice
```

If OCR tools are not installed, the app can still run, but extraction for scanned PDFs / legacy docs may be limited.
