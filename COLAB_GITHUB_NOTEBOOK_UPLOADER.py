from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import quote

import ipywidgets as widgets
import requests
from google.colab import drive, output
from IPython.display import HTML, display

REPOSITORY = "gear66me-ui/IERS_TN36_V01_MASTER-1"
BRANCH = "main"
TARGET_DEFAULT = "IERS_TN36_V02_MASTER.ipynb"
DRIVE_ROOT = Path("/content/drive/MyDrive")

output.enable_custom_widget_manager()
drive.mount("/content/drive", force_remount=False)

notebook = widgets.Dropdown(
    description="Notebook:",
    layout=widgets.Layout(width="100%"),
    style={"description_width": "95px"},
)
target = widgets.Text(
    value=TARGET_DEFAULT,
    description="GitHub file:",
    layout=widgets.Layout(width="100%"),
    style={"description_width": "95px"},
)
token = widgets.Password(
    placeholder="Used only if GITHUB_TOKEN is blank",
    description="Token:",
    layout=widgets.Layout(width="100%"),
    style={"description_width": "95px"},
)
refresh = widgets.Button(description="Refresh", icon="refresh", button_style="info")
upload = widgets.Button(
    description="Upload to GitHub",
    icon="cloud-upload",
    button_style="success",
    layout=widgets.Layout(width="230px"),
)
status = widgets.Output(
    layout={"border": "1px solid #888", "padding": "8px", "margin": "8px 0 0 0"}
)


def configured_token() -> str:
    return os.environ.get("GITHUB_TOKEN", "").strip() or token.value.strip()


def scan_notebooks() -> list[tuple[str, str]]:
    found: list[tuple[float, str, str]] = []
    preferred = DRIVE_ROOT / "Colab Notebooks"
    roots = [preferred] if preferred.exists() else [DRIVE_ROOT]

    for search_root in roots:
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [name for name in dirs if name != ".ipynb_checkpoints"]
            for name in files:
                if not name.lower().endswith(".ipynb"):
                    continue
                path = Path(root) / name
                try:
                    stat = path.stat()
                    relative = path.relative_to(DRIVE_ROOT)
                    label = f"{relative}  [{stat.st_size / 1048576:.3f} MiB]"
                    found.append((stat.st_mtime, label, str(path)))
                except OSError:
                    continue

    found.sort(reverse=True)
    return [(label, path) for _, label, path in found[:200]]


def refresh_list(_=None) -> None:
    with status:
        status.clear_output()
        print("Scanning Google Drive...")

    options = scan_notebooks()
    notebook.options = options

    with status:
        status.clear_output()
        if options:
            notebook.value = options[0][1]
            print(f"Found {len(options)} notebook(s).")
            print(f"Newest: {options[0][0]}")
            if os.environ.get("GITHUB_TOKEN", "").strip():
                print("GitHub token loaded from the Colab cell.")
            else:
                print("GitHub token is blank.")
        else:
            print("No .ipynb files found in My Drive/Colab Notebooks.")


def github_error(response: requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("message", body))
    except Exception:
        return response.text[:800]


def upload_notebook(_=None) -> None:
    with status:
        status.clear_output()

        source_text = notebook.value
        target_path = target.value.strip().lstrip("/")
        secret = configured_token()

        if not source_text:
            print("ERROR: Select a notebook.")
            return
        if not target_path.lower().endswith(".ipynb"):
            print("ERROR: GitHub filename must end in .ipynb.")
            return
        if not secret:
            print('ERROR: Put your token between the quotes in GITHUB_TOKEN="".')
            return

        source = Path(source_text)
        if not source.exists():
            print(f"ERROR: File not found: {source}")
            return

        try:
            raw = source.read_bytes()
            json.loads(raw.decode("utf-8"))
        except Exception as exc:
            print(f"ERROR: Invalid notebook JSON: {exc}")
            return

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {secret}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        api = (
            f"https://api.github.com/repos/{REPOSITORY}/contents/"
            f"{quote(target_path, safe='/')}"
        )

        print(f"Uploading {source.name} ({len(raw) / 1048576:.3f} MiB)...")

        try:
            existing = requests.get(
                api,
                headers=headers,
                params={"ref": BRANCH},
                timeout=60,
            )
        except requests.RequestException as exc:
            print(f"ERROR contacting GitHub: {exc}")
            return

        payload = {
            "message": f"Upload {Path(target_path).name} from Google Colab",
            "branch": BRANCH,
            "content": base64.b64encode(raw).decode("ascii"),
        }

        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]
        elif existing.status_code != 404:
            print(f"ERROR {existing.status_code}: {github_error(existing)}")
            token.value = ""
            return

        try:
            response = requests.put(
                api,
                headers=headers,
                json=payload,
                timeout=300,
            )
        except requests.RequestException as exc:
            print(f"ERROR during upload: {exc}")
            return
        finally:
            token.value = ""

        if response.status_code not in (200, 201):
            print(f"ERROR {response.status_code}: {github_error(response)}")
            return

        data = response.json()
        github_url = data.get("content", {}).get("html_url", "")
        colab_url = (
            "https://colab.research.google.com/github/"
            f"{REPOSITORY}/blob/{BRANCH}/{target_path}"
        )

        print("SUCCESS — notebook committed to GitHub.")
        if github_url:
            display(
                HTML(
                    f'<a href="{github_url}" target="_blank">'
                    "<b>Open on GitHub</b></a>"
                )
            )
        display(
            HTML(
                f'<br><a href="{colab_url}" target="_blank">'
                "<b>Open GitHub-backed notebook in Colab</b></a>"
            )
        )


refresh.on_click(refresh_list)
upload.on_click(upload_notebook)

token_loaded = bool(os.environ.get("GITHUB_TOKEN", "").strip())
token_notice = widgets.HTML(
    "<b>Token:</b> loaded from the Colab cell."
    if token_loaded
    else "<b>Token:</b> blank; enter it above or in the loader cell."
)

items = [
    widgets.HTML("<h3>Drive → GitHub Notebook Uploader</h3>"),
    refresh,
    notebook,
    target,
    token_notice,
]
if not token_loaded:
    items.append(token)
items.extend([upload, status])

display(widgets.VBox(items))
refresh_list()
