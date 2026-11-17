"""Publish local array bundles under immutable IDs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray


def _publish_bundle(bundle: Path, destination: Path) -> None:
    # Windows file indexing can hold a just-written file open without delete sharing.
    for delay in (0.02, 0.05, 0.10, 0.20, 0.40):
        try:
            bundle.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or destination.exists():
                raise
            sleep(delay)
    bundle.rename(destination)


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, artifact_id: str) -> Path:
        return self.root / UUID(artifact_id).hex

    def save(
        self,
        arrays: dict[str, NDArray],
        metadata: dict,
        *,
        preview_png: bytes | None = None,
        previews: dict[str, bytes] | None = None,
        attachments: dict[str, bytes] | None = None,
    ) -> str:
        artifact_id = uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".staging-", dir=self.root) as staging:
            bundle = Path(staging) / "bundle"
            bundle.mkdir()
            entries = {}
            for name, values in arrays.items():
                if not name.isidentifier():
                    raise ValueError(f"invalid array name {name!r}")
                filename = f"{name}.npy"
                np.save(bundle / filename, values, allow_pickle=False)
                entries[name] = {"file": filename, "dtype": str(values.dtype), "shape": list(values.shape)}
            if preview_png is not None:
                (bundle / "preview.png").write_bytes(preview_png)
            preview_entries = {}
            for name, payload in (previews or {}).items():
                if not name.isidentifier() or name == "preview":
                    raise ValueError(f"invalid additional preview name {name!r}")
                filename = f"{name}.png"
                (bundle / filename).write_bytes(payload)
                preview_entries[name] = filename
            manifest = {
                **metadata,
                "artifact_id": artifact_id,
                "arrays": entries,
                "preview": "preview.png" if preview_png is not None else None,
            }
            if preview_entries:
                manifest["previews"] = preview_entries
            if attachments:
                for name, payload in attachments.items():
                    attachment = Path(name)
                    if attachment.name != name or not attachment.stem.isidentifier() or attachment.suffix not in (".html", ".csv"):
                        raise ValueError(f"invalid report attachment name {name!r}")
                    (bundle / name).write_bytes(payload)
                manifest["attachments"] = list(attachments)
            (bundle / "metadata.json").write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
            )
            _publish_bundle(bundle, self.path(artifact_id))
        return artifact_id

    def load_metadata(self, artifact_id: str) -> dict:
        with (self.path(artifact_id) / "metadata.json").open(encoding="utf-8") as manifest:
            return json.load(manifest)

    def load_array(self, artifact_id: str, name: str, *, mmap_mode: str | None = None) -> NDArray:
        if not name.isidentifier():
            raise ValueError(f"invalid array name {name!r}")
        return np.load(self.path(artifact_id) / f"{name}.npy", allow_pickle=False, mmap_mode=mmap_mode)
