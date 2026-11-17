"""Browse published artifacts without exposing files outside the artifact store."""

import json
from pathlib import Path
from uuid import UUID

import streamlit as st

from src.storage.artifact_store import ArtifactStore

PROJECT = Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    return Path("artifacts").resolve()


@st.cache_data(max_entries=256, show_spinner=False)
def _metadata(path: Path, modified_ns: int) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def catalog(folder: str, *, artifact_type: str | None = None, limit: int | None = None) -> dict[str, dict]:
    directory = artifact_root() / folder
    if not directory.is_dir():
        return {}
    entries = []
    for bundle in directory.iterdir():
        try:
            identifier = UUID(bundle.name).hex
            if identifier == bundle.name:
                path = bundle / "metadata.json"
                entries.append((path.stat().st_mtime_ns, identifier, path))
        except (OSError, ValueError):
            continue
    rows = {}
    for stamp, identifier, path in sorted(entries, reverse=True):
        try:
            item = _metadata(path, stamp)
            if item.get("artifact_id") == identifier and (artifact_type is None or item.get("artifact_type") == artifact_type):
                rows[identifier] = item
                if limit is not None and len(rows) >= limit:
                    break
        except (OSError, ValueError):
            continue
    return rows


def choose_artifact(label: str, rows: dict[str, dict], *, key: str, names=None) -> str | None:
    if not rows:
        st.info("当前没有可用产物。")
        return None
    return st.selectbox(label, list(rows), key=key,
                        format_func=(lambda identifier: names(rows[identifier], identifier)) if names else None)


def burst_label(meta: dict, identifier: str) -> str:
    kind = "模拟" if meta.get("artifact_type") == "synthetic_raw_burst" else "实拍"
    kind = meta.get("source", {}).get("scene_label") or kind
    return f'{kind} · {len(meta["frames"])} 帧 · {identifier[:12]}'


def final_download(folder: str, identifier: str, *, key: str, show: bool = True) -> None:
    path = ArtifactStore(artifact_root() / folder).path(identifier) / "preview.png"
    if not path.is_file():
        st.warning("图像文件未在本地产物库中找到。")
        return
    preview = path.read_bytes()
    if show:
        st.image(preview, alt="最终处理结果", width="stretch")
    st.download_button("下载最终 PNG", preview, f"aicpl_{identifier}.png", "image/png",
                       key=key, on_click="ignore", icon=":material/download:")
