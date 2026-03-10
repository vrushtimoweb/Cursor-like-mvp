from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import os
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_WORKSPACE = Path.home().resolve()
CURRENT_WORKSPACE = DEFAULT_WORKSPACE

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".next",
    ".cache",
}

IGNORED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA2Q5uVeyJ51A4H0z1Z8GishjJG1CSeo2A")
client = None
if OpenAI and GEMINI_API_KEY:
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


class FileRequest(BaseModel):
    path: str


class WriteRequest(BaseModel):
    path: str
    content: str


class ChatRequest(BaseModel):
    instruction: str
    code: str
    file_path: str | None = None


class RunRequest(BaseModel):
    path: str


class OpenProjectRequest(BaseModel):
    folder_path: str


def get_workspace() -> Path:
    global CURRENT_WORKSPACE
    return CURRENT_WORKSPACE


def is_ignored_file(path: Path) -> bool:
    return path.suffix.lower() in IGNORED_FILE_SUFFIXES


def safe_path(relative_path: str) -> Path:
    workspace = get_workspace()
    full_path = (workspace / relative_path).resolve()
    if not str(full_path).startswith(str(workspace)):
        raise ValueError("Invalid path")
    return full_path


def build_file_tree(root: Path, current: Path) -> list[dict]:
    items = []

    try:
        entries = sorted(
            current.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        return items

    for entry in entries:
        if entry.name in IGNORED_DIRS:
            continue

        if entry.is_file() and is_ignored_file(entry):
            continue

        rel_path = str(entry.relative_to(root))

        if entry.is_dir():
            items.append(
                {
                    "type": "directory",
                    "name": entry.name,
                    "path": rel_path,
                    "children": build_file_tree(root, entry),
                }
            )
        else:
            items.append(
                {
                    "type": "file",
                    "name": entry.name,
                    "path": rel_path,
                }
            )

    return items


def list_all_files(root: Path) -> list[str]:
    results = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file_name in files:
            file_path = Path(current_root) / file_name
            if is_ignored_file(file_path):
                continue
            results.append(str(file_path.relative_to(root)))

    return sorted(results)


def pick_folder_native() -> str | None:
    # Linux GUI picker via zenity if available
    if shutil.which("zenity"):
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", "--title=Open Project Folder"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            selected = result.stdout.strip()
            return selected or None

    # Fallback: tkinter dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Open Project Folder")
    root.destroy()

    return selected or None


@app.get("/project")
def get_project():
    workspace = get_workspace()
    return {
        "root": str(workspace),
        "name": workspace.name,
    }


@app.post("/open-project")
def open_project(data: OpenProjectRequest):
    global CURRENT_WORKSPACE

    folder = Path(data.folder_path).expanduser().resolve()

    if not folder.exists():
        raise HTTPException(status_code=400, detail="Folder does not exist")

    if not folder.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a folder")

    CURRENT_WORKSPACE = folder

    return {
        "status": "opened",
        "root": str(CURRENT_WORKSPACE),
        "name": CURRENT_WORKSPACE.name,
    }


@app.post("/pick-project")
def pick_project():
    global CURRENT_WORKSPACE

    selected = pick_folder_native()

    if not selected:
        raise HTTPException(status_code=400, detail="No folder selected")

    folder = Path(selected).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Selected path is not a valid folder")

    CURRENT_WORKSPACE = folder

    return {
        "status": "opened",
        "root": str(CURRENT_WORKSPACE),
        "name": CURRENT_WORKSPACE.name,
    }


@app.get("/files")
def list_files():
    workspace = get_workspace()
    return {"files": list_all_files(workspace)}


@app.get("/tree")
def get_tree():
    workspace = get_workspace()
    return {
        "root": str(workspace),
        "name": workspace.name,
        "tree": build_file_tree(workspace, workspace),
    }


@app.post("/read")
def read_file(data: FileRequest):
    try:
        path = safe_path(data.path)
        return {"content": path.read_text(encoding="utf-8")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/write")
def write_file(data: WriteRequest):
    try:
        path = safe_path(data.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data.content, encoding="utf-8")
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
def chat(data: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini client not configured. Set GEMINI_API_KEY in your environment.",
        )

    system_prompt = (
        "You are an AI coding assistant inside an IDE. "
        "Return only updated code unless the user explicitly asks for explanation. "
        "Do not wrap the answer in markdown fences."
    )

    user_prompt = f"""
Instruction:
{data.instruction}

File:
{data.file_path or "unknown"}

Current code:
{data.code}
"""

    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return {"response": response.choices[0].message.content or ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
def run_file(data: RunRequest):
    try:
        path = safe_path(data.path)
        result = subprocess.run(
            ["python3", str(path)],
            capture_output=True,
            text=True,
            cwd=str(get_workspace()),
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))