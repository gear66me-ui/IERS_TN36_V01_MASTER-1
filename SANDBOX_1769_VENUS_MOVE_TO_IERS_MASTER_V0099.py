# V0099
# Audit reference: GitHub migration widget; copies 1769 Venus transit files from GitHub_Sandbox to IERS_TN36_V01_MASTER-1 using GitHub API.
from __future__ import annotations

import base64
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def req(name: str, pkg: str) -> None:
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", pkg])


req("requests", "requests")
import requests

VERSION = "V0099"
TZ = ZoneInfo("America/Bogota")
SOURCE_REPO = "gear66me-ui/GitHub_Sandbox"
DEST_REPO = "gear66me-ui/IERS_TN36_V01_MASTER-1"
BRANCH = "main"
DELETE_SOURCE_AFTER_COPY = False
FILES = [
    "PYTHON/VENUS_1769_V0027_FORMAT_STANDALONE_V0067.py",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0068_WIDGET.py",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0069_WIDGET.py",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0070_WIDGET.py",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0071_WIDGET.py",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0072_WIDGET.py",
    "VENUS_1769_V0027_FORMAT_STANDALONE_V0080.py",
    "v0070_chunks/chunk_00.txt",
]


def get_token() -> str:
    names = ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT", "PAT")
    for name in names:
        val = os.environ.get(name)
        if val:
            return val.strip()
    try:
        from google.colab import userdata
        for name in names:
            try:
                val = userdata.get(name)
                if val:
                    return str(val).strip()
            except Exception:
                pass
    except Exception:
        pass
    raise RuntimeError(
        "GitHub token not found. Add a Colab secret named GITHUB_TOKEN, GH_TOKEN, GITHUB_PAT, or PAT."
    )


def api_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "IERS-TN36-Venus-Migration-V0099",
    }


def raw_url(repo: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{BRANCH}/{path}"


def contents_url(repo: str, path: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents/{path}"


def fetch_source_bytes(path: str) -> bytes:
    url = raw_url(SOURCE_REPO, path)
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"FETCH FAILED {path}: HTTP {r.status_code} {r.text[:240]}")
    return r.content


def destination_sha(token: str, path: str) -> str | None:
    r = requests.get(contents_url(DEST_REPO, path), headers=api_headers(token), params={"ref": BRANCH}, timeout=120)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"DEST SHA FAILED {path}: HTTP {r.status_code} {r.text[:240]}")
    data = r.json()
    if isinstance(data, dict):
        return data.get("sha")
    raise RuntimeError(f"DEST SHA FAILED {path}: destination is not a file")


def put_destination(token: str, path: str, content: bytes) -> str:
    sha = destination_sha(token, path)
    payload = {
        "message": f"Move sandbox 1769 Venus file: {path}",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(contents_url(DEST_REPO, path), headers=api_headers(token), json=payload, timeout=120)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"PUT FAILED {path}: HTTP {r.status_code} {r.text[:500]}")
    return r.json()["commit"]["sha"]


def delete_source(token: str, path: str) -> str:
    r = requests.get(contents_url(SOURCE_REPO, path), headers=api_headers(token), params={"ref": BRANCH}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"SOURCE SHA FAILED {path}: HTTP {r.status_code} {r.text[:240]}")
    sha = r.json()["sha"]
    payload = {"message": f"Remove migrated 1769 Venus file: {path}", "sha": sha, "branch": BRANCH}
    d = requests.delete(contents_url(SOURCE_REPO, path), headers=api_headers(token), json=payload, timeout=120)
    if d.status_code not in (200, 204):
        raise RuntimeError(f"DELETE FAILED {path}: HTTP {d.status_code} {d.text[:500]}")
    try:
        return d.json()["commit"]["sha"]
    except Exception:
        return "DELETE_OK"


def main() -> None:
    print("CODE INPUTS")
    print(f"Version: {VERSION}")
    print(f"Source repo: {SOURCE_REPO}")
    print(f"Destination repo: {DEST_REPO}")
    print(f"Branch: {BRANCH}")
    print(f"Files requested: {len(FILES)}")
    print(f"Delete source after copy: {DELETE_SOURCE_AFTER_COPY}")
    print("COMMENTS")
    print("Copies the 1769 Venus transit project files from GitHub_Sandbox into the IERS master repository.")
    print("Large compressed Python payload files are fetched through raw GitHub bytes, not pasted manually.")
    print("Source deletion is disabled for safety; this is a verified migration/copy stage.")
    token = get_token()
    print("RESULTS")
    copied = []
    for path in FILES:
        data = fetch_source_bytes(path)
        commit = put_destination(token, path, data)
        copied.append((path, len(data), commit))
        print(f"COPIED | {len(data):9d} bytes | {path} | commit {commit[:12]}")
        if DELETE_SOURCE_AFTER_COPY:
            dcommit = delete_source(token, path)
            print(f"DELETED SOURCE | {path} | commit {dcommit[:12]}")
    print("OUTPUT SUMMARY")
    print(f"Copied files: {len(copied)}")
    print(f"Copied bytes: {sum(x[1] for x in copied)}")
    print("PAPER COMPARISON")
    print("NOT USED — repository migration only; no scientific quantities recalculated.")
    print("EQUATION STATUS")
    print("NOT USED — no equations executed; files copied byte-for-byte from source raw URLs.")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(VERSION)


if __name__ == "__main__":
    main()
# V0099
