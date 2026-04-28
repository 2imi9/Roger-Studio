from pathlib import Path

from fastapi import APIRouter, HTTPException, Response, UploadFile

from app.models.schemas import DatasetInfo
from app.services.data_ingest import (
    UPLOAD_DIR,
    DataFormat,
    _NO_MAGIC_EXTS,
    detect_format,
    inspect_file,
    sniff_format_by_magic,
)
from app.services import database as db
from app.services.raster_tiles import render_geotiff_tile

router = APIRouter()

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB


def _safe_upload_path(filename: str) -> Path:
    """Resolve ``UPLOAD_DIR / filename`` and refuse anything that escapes
    ``UPLOAD_DIR``.

    Closes a path-traversal hole: the frontend calls
    ``encodeURIComponent(filename)`` on its side, but FastAPI decodes path
    parameters + ``UploadFile.filename`` before handing them to us, so a
    caller (or anyone hitting the API directly) could pass
    ``"../etc/passwd"`` or an absolute Windows path like ``"C:/secret"``
    and land read/write operations outside the intended upload directory.

    Uses ``.resolve()`` to collapse ``..`` / symlinks, then
    ``.relative_to(UPLOAD_DIR)`` which raises ``ValueError`` if the path
    is outside the tree. We map that into an HTTP 400 so the frontend
    sees a proper rejection rather than a 500.

    Additionally rejects empty filenames + any that contain a path
    separator (``/`` or ``\\``) or null byte — those are always
    suspicious and never produced by the happy-path upload flow.
    """
    if not filename or "\x00" in filename:
        raise HTTPException(400, "invalid filename")
    # Any path separator in the leaf is suspect: uploads should produce a
    # single-segment name. This catches ``..\\escape`` and ``a/b`` cases
    # before pathlib's resolve() even runs.
    if "/" in filename or "\\" in filename:
        raise HTTPException(400, "invalid filename (path separators not allowed)")
    root = UPLOAD_DIR.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise HTTPException(400, f"filename escapes upload directory: {filename!r}") from e
    return candidate


@router.post("/upload", response_model=DatasetInfo)
async def upload_file(file: UploadFile) -> DatasetInfo:
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    fmt = detect_format(file.filename)
    if fmt == DataFormat.UNKNOWN:
        raise HTTPException(
            400,
            f"Unsupported format: {file.filename}. "
            "Supported: GeoTIFF, NetCDF, GeoPackage, GeoJSON, Shapefile(.zip), "
            "GeoParquet, CSV",
        )

    # Save to temp — _safe_upload_path rejects any filename that resolves
    # outside UPLOAD_DIR (path-traversal guard).
    dest = _safe_upload_path(file.filename)

    # Peek the first 16 bytes before writing anything so we can reject an
    # extension/content mismatch *before* committing a large file to disk.
    # The audit caught this: previously `detect_format` trusted only the
    # extension, so `evil.geojson` containing raw TIFF bytes would be
    # written, inspected as vector, and fail deep inside the ingest stack
    # with a confusing error (or worse, be served to downstream vector
    # tools). Magic-byte sniffing catches the lie at the door.
    peek = await file.read(16)
    sniffed = sniff_format_by_magic(peek)
    ext = Path(file.filename).suffix.lower()
    if sniffed and fmt not in sniffed and ext not in _NO_MAGIC_EXTS:
        sniffed_names = sorted(f.value for f in sniffed)
        raise HTTPException(
            400,
            f"Format mismatch: filename suggests {fmt.value!r} but file "
            f"contents look like {sniffed_names!r}. Rename the file to "
            "match its actual format, or re-export from the source tool.",
        )

    size = len(peek)
    with open(dest, "wb") as f:
        f.write(peek)
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File too large (max 500MB)")
            f.write(chunk)

    # Inspect
    try:
        info = inspect_file(str(dest), file.filename)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Failed to read file: {e}")

    # Persist to SQLite
    db.save_dataset(info, str(dest))
    return info


@router.get("/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    return db.list_datasets()


@router.get("/datasets/{filename}", response_model=DatasetInfo)
async def get_dataset(filename: str) -> DatasetInfo:
    info = db.get_dataset(filename)
    if info is None:
        raise HTTPException(404, f"Dataset '{filename}' not found")
    return info


@router.delete("/datasets/{filename}")
async def delete_dataset(filename: str) -> dict:
    # Remove file — path-traversal guarded so a malicious filename can't
    # delete arbitrary files on disk via DELETE /datasets/../etc/passwd.
    dest = _safe_upload_path(filename)
    dest.unlink(missing_ok=True)
    if not db.delete_dataset(filename):
        raise HTTPException(404, f"Dataset '{filename}' not found")
    return {"deleted": filename}


@router.get("/datasets/{filename}/tiles/{z}/{x}/{y}.png")
async def dataset_tile(filename: str, z: int, x: int, y: int) -> Response:
    """Serve one XYZ tile for an uploaded GeoTIFF.

    Lets the frontend drop uploaded rasters onto the map as a layer —
    feeding the Compare view for side-by-side inspection against an
    OlmoEarth inference output over the same bbox.
    """
    info = db.get_dataset(filename)
    if info is None:
        raise HTTPException(404, f"Dataset '{filename}' not found")
    # Path-traversal guard even on tile reads: attackers could use the
    # tile endpoint as an oracle for arbitrary file existence otherwise.
    path = _safe_upload_path(filename)
    if not path.exists():
        raise HTTPException(410, f"Dataset '{filename}' file is missing on disk")
    # Defer any heavy detection to render_geotiff_tile — it's only PNG work.
    try:
        png = render_geotiff_tile(path, z, x, y)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(422, f"Failed to render tile: {e}") from e
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "ETag": f'"{filename}-{z}-{x}-{y}"',
        },
    )
