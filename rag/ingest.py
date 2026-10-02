"""Document ingestion: loading, chunking, partitioning, ZIP handling."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rag.config import AUTO_PARTITION_MAX_CORPUS_CHARS, CHUNK_OVERLAP, CHUNK_WORDS, GENERIC_FOLDER_NAMES, MAX_ZIP_FILES, MAX_ZIP_MEMBER_BYTES, SUPPORTED_EXTENSIONS, UPLOAD_DIR
from rag.text import STOPWORDS, clean_text, estimate_tokens, extract_entities, stem_token
from rag.db import ChunkMetadata, DatabaseManager


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@dataclass
class IngestionSummary:
    processed: int = 0
    duplicates: List[str] = None
    skipped: List[str] = None
    failures: List[str] = None
    partitions: Dict[str, int] = None

    def __post_init__(self) -> None:
        self.duplicates = self.duplicates or []
        self.skipped = self.skipped or []
        self.failures = self.failures or []
        self.partitions = self.partitions or {}

    def as_markdown(self) -> str:
        lines = [f"**Ingestion complete:** {self.processed} file(s) indexed."]
        if self.partitions:
            lines.append("Partitions: " + ", ".join(f"`{name.replace('partition_', '')}`: {count}" for name, count in sorted(self.partitions.items())))
        if self.duplicates:
            lines.append("**Duplicates ignored:** " + ", ".join(self.duplicates))
        if self.skipped:
            lines.append("**Unsupported/skipped:** " + ", ".join(self.skipped))
        if self.failures:
            lines.append("**Could not ingest:** " + "; ".join(self.failures))
        return "\n\n".join(lines)


def _hash_file_streaming(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file's contents without reading it into memory at
    once -- matters once files (or the zips containing them) get into
    the hundreds of MB / low GB range."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


class DocumentProcessor:
    def __init__(self, db: DatabaseManager, lazy: bool = False):
        self.db = db
        self.lazy = lazy
        if lazy:
            # Large-corpus mode: don't load every chunk's text/metadata into
            # RAM up front. ComparisonEngine queries the database directly
            # (FTS5 + chunk_entities) instead of these dicts when lazy=True;
            # they're kept as empty dicts rather than removed so any code
            # that still reads processor.chunk_texts as a dict doesn't crash,
            # it just correctly sees nothing resident (use db methods for
            # actual lookups instead).
            self.chunk_texts, self.chunks_metadata = {}, {}
        else:
            self.chunk_texts, self.chunks_metadata = self.db.get_all_chunks()

    @staticmethod
    def _load_document(file_path: str) -> str:
        path = Path(file_path.strip().strip('"\''))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
            except ImportError as exc:
                raise ImportError("PDF support requires: pip install pymupdf") from exc
            with fitz.open(path) as pdf:
                return clean_text("\n".join(page.get_text() for page in pdf))
        if suffix in {".txt", ".md", ".py", ".js"}:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                cells = notebook.get("cells", [])
                return clean_text("\n\n".join("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", "")) for cell in cells if cell.get("cell_type") in {"markdown", "code"}))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid notebook JSON: {path.name}") from exc
        raise ValueError(f"Unsupported file type: {suffix}")

    _CITATION_MARKER_RE = re.compile(r"\[\d{1,3}(?:,\s*\d{1,3})*\]|\(\d{4}[a-z]?\)|et al\.")

    @classmethod
    def _is_non_content_chunk(cls, text: str) -> bool:
        opening = text[:180].lower().strip()
        if (
            len(text.split()) < 25
            or opening.startswith("references")
            or opening.startswith("bibliography")
            or opening.startswith("acknowledg")
        ):
            return True
        words = text.split()
        if not words:
            return True
        markers = cls._CITATION_MARKER_RE.findall(text)
        citation_density_per_100_words = (len(markers) / max(1, len(words))) * 100
        if citation_density_per_100_words >= 12.0:
            return True
        return False

    def _classify_document(self, text: str, file_path: str) -> str:
        """
        Assigns a partition ID based on the immediate parent folder of the
        file. If the file is at the root, or its folder name carries no
        signal (e.g. "documents", "uploads"), partition = 'general'.
        """
        path = Path(file_path)
        parent = path.parent.name
        if parent and parent != "." and parent.lower() not in GENERIC_FOLDER_NAMES:
            return f"partition_{parent}"
        return "partition_general"

    def _keyword_label(self, text: str, top_n: int = 2) -> str:
        """Cheap, corpus-agnostic label for a single document: its own
        most frequent non-trivial words. Needs only the document's own
        text (no corpus-wide stats), so it's safe to use per-file even
        for very large corpora where clustering the whole corpus at once
        isn't practical.
        """
        window = text[:4000].lower()
        tokens = [
            stem_token(token) for token in re.findall(r"\b[a-z][a-z-]{3,}\b", window)
            if token not in STOPWORDS
        ]
        if not tokens:
            return "general"
        top = [term for term, _ in Counter(tokens).most_common(top_n)]
        return "_".join(top) or "general"

    # --------------------------------------------------------------------
    # NEW: Auto‑partitioning via content clustering (dump‑and‑go)
    # --------------------------------------------------------------------
    def _auto_partition_documents(self, file_paths: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Automatically assign partition IDs for a flat, un-foldered corpus.
        Returns (partition_map, preloaded_texts) -- texts are handed back
        so process_uploads() can pass them straight into ingestion instead
        of re-parsing every PDF/notebook a second time, which matters once
        a corpus gets into the hundreds of MB.

        Falls back to a cheap per-document keyword label (no clustering,
        no corpus held in memory at once) when scikit-learn isn't
        installed, or when the corpus is too large to cluster cheaply --
        this is what keeps a ~1.5GB upload fast instead of stalling on a
        single giant TF-IDF/KMeans pass.
        """
        texts: Dict[str, str] = {}
        total_chars = 0
        for fp in file_paths:
            try:
                text = self._load_document(fp)
            except Exception:
                continue
            if text and len(text.split()) > 50:  # ignore very short docs
                texts[fp] = text
                total_chars += len(text)

        if not texts:
            return {fp: "partition_general" for fp in file_paths}, {}

        def _keyword_fallback() -> Dict[str, str]:
            mapping = {fp: f"partition_{self._keyword_label(text)}" for fp, text in texts.items()}
            for fp in file_paths:
                mapping.setdefault(fp, "partition_general")
            return mapping

        if total_chars > AUTO_PARTITION_MAX_CORPUS_CHARS:
            print(
                f"Corpus text ({total_chars:,} chars) exceeds AUTO_PARTITION_MAX_CORPUS_CHARS "
                f"({AUTO_PARTITION_MAX_CORPUS_CHARS:,}); using per-document keyword "
                "partitioning instead of whole-corpus clustering."
            )
            return _keyword_fallback(), texts

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.decomposition import TruncatedSVD
            import numpy as np
        except ImportError:
            print(
                "scikit-learn not installed; using per-document keyword partitioning "
                "(pip install scikit-learn for content-clustered partitions)."
            )
            return _keyword_fallback(), texts

        valid_paths = list(texts.keys())
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(texts.values())

        n_components = min(100, X.shape[1] - 1, len(valid_paths) - 1)
        if n_components < 2:
            partition_map = {fp: "partition_general" for fp in valid_paths}
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X)
            n_docs = len(valid_paths)
            n_clusters = min(max(2, int(np.sqrt(n_docs))), 10, n_docs)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            partition_map = {fp: f"partition_cluster_{label}" for fp, label in zip(valid_paths, labels)}
        for fp in file_paths:
            partition_map.setdefault(fp, "partition_general")
        return partition_map, texts

    def _chunk_document(
        self, text: str, source_path: str, partition_id: str
    ) -> Tuple[Dict[str, str], Dict[str, ChunkMetadata]]:
        words = clean_text(text).split()
        chunks: Dict[str, str] = {}
        metadata: Dict[str, ChunkMetadata] = {}
        source_hash = hashlib.sha1(str(Path(source_path).resolve()).encode("utf-8")).hexdigest()[:12]
        step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
        chunk_number = 0
        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_WORDS]
            if len(chunk_words) < 25:
                continue
            chunk_text = " ".join(chunk_words)
            if self._is_non_content_chunk(chunk_text):
                continue
            chunk_number += 1
            chunk_id = f"{source_hash}_{chunk_number:04d}"
            entities = extract_entities(chunk_text)
            chunks[chunk_id] = chunk_text
            metadata[chunk_id] = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=str(Path(source_path).resolve()),
                partition_id=partition_id,
                chunk_index=chunk_number,
                token_count=estimate_tokens(chunk_text),
                entity_count=len(entities),
            )
            if start + CHUNK_WORDS >= len(words):
                break
        return chunks, metadata

    def _extract_zip(self, archive_path: Path, summary: IngestionSummary) -> Tuple[List[Path], Set[str]]:
        """Extract supported members, preserving the archive's internal
        folder structure on disk (so hr/policy.md stays under an "hr"
        folder after extraction -- this is what lets _classify_document's
        folder-based partitioning pick it up unchanged).

        Also returns the set of extracted paths that sat directly at the
        archive's root with no internal subfolder. Those files' on-disk
        parent will just be this archive's own randomly-named extraction
        destination -- a real signal for nothing -- so callers must not
        mistake it for a curated category folder the way a genuine
        internal subfolder (hr/, legal/, ...) is.
        """
        try:
            if not zipfile.is_zipfile(archive_path):
                summary.failures.append(f"{archive_path.name} (invalid ZIP archive)")
                return [], set()
            archive_hash = _hash_file_streaming(archive_path)[:16]
            destination = (UPLOAD_DIR / f"{archive_path.stem}_{archive_hash}").resolve()
            destination.mkdir(parents=True, exist_ok=True)
            extracted: List[Path] = []
            flat_root_paths: Set[str] = set()
            with zipfile.ZipFile(archive_path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if len(members) > MAX_ZIP_FILES:
                    summary.failures.append(f"{archive_path.name} (contains {len(members)} files; limit is {MAX_ZIP_FILES})")
                    return [], set()
                for member in members:
                    member_path = Path(member.filename)
                    suffix = member_path.suffix.lower()
                    if suffix not in SUPPORTED_EXTENSIONS:
                        summary.skipped.append(f"{member.filename} (unsupported {suffix or 'file type'})")
                        continue
                    if member.file_size > MAX_ZIP_MEMBER_BYTES:
                        summary.skipped.append(f"{member.filename} (larger than {MAX_ZIP_MEMBER_BYTES // (1024 * 1024)} MB limit)")
                        continue
                    target = (destination / member_path).resolve()
                    if destination not in target.parents:
                        summary.skipped.append(f"{member.filename} (unsafe archive path)")
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        output.write(source.read())
                    extracted.append(target)
                    if member_path.parent == Path("."):
                        flat_root_paths.add(str(target))
            return extracted, flat_root_paths
        except (OSError, zipfile.BadZipFile) as exc:
            summary.failures.append(f"{archive_path.name} ({exc})")
            return [], set()

    # --------------------------------------------------------------------
    # MODIFIED: _process_document now accepts partition_map
    # --------------------------------------------------------------------
    def _process_document(self, path: Path, existing_hashes: Dict[str, str], force: bool,
                          summary: IngestionSummary, partition_map: Dict[str, str] = None,
                          preloaded_text: Optional[str] = None) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            summary.skipped.append(f"{path.name} (unsupported {path.suffix or 'file type'})")
            return
        if not path.exists():
            summary.failures.append(f"{path.name} (file not found)")
            return
        canonical_path = str(path.resolve())
        try:
            content_hash = _hash_file_streaming(path)
            if not force and content_hash in existing_hashes:
                summary.duplicates.append(f"{path.name} (same as {Path(existing_hashes[content_hash]).name})")
                return
            text = preloaded_text if preloaded_text is not None else self._load_document(canonical_path)
            if not text:
                summary.skipped.append(f"{path.name} (empty)")
                return
            # Determine partition: if partition_map provided, use it; else fallback to folder‑based
            if partition_map and canonical_path in partition_map:
                partition = partition_map[canonical_path]
            else:
                partition = self._classify_document(text, canonical_path)
            chunks, metadata = self._chunk_document(text, canonical_path, partition)
            if not chunks:
                summary.skipped.append(f"{path.name} (no usable content chunks)")
                return
            if not self.lazy:
                old_chunk_ids = [chunk_id for chunk_id, old_metadata in self.chunks_metadata.items() if old_metadata.file_path == canonical_path]
                for chunk_id in old_chunk_ids:
                    self.chunk_texts.pop(chunk_id, None)
                    self.chunks_metadata.pop(chunk_id, None)
            self.db.save_document(canonical_path, partition, chunks, metadata, content_hash=content_hash)
            if not self.lazy:
                self.chunk_texts.update(chunks)
                self.chunks_metadata.update(metadata)
            existing_hashes[content_hash] = canonical_path
            summary.processed += 1
            summary.partitions[partition] = summary.partitions.get(partition, 0) + 1
        except Exception as exc:
            summary.failures.append(f"{path.name} ({exc})")

    # --------------------------------------------------------------------
    # MODIFIED: process_uploads with auto‑partitioning trigger
    # --------------------------------------------------------------------
    def process_uploads(self, file_paths: Sequence[str], force: bool = False) -> IngestionSummary:
        """Ingest direct files and ZIP archives, with auto-partitioning if needed.

        ZIPs are extracted first so the partitioning decision is made
        against the files that will actually be indexed -- including a
        ZIP's own internal folder structure (e.g. hr/, legal/, ...), which
        is used as-is when it exists. Content-based clustering only
        kicks in for genuinely flat corpora (no folder signal at all,
        whether uploaded directly or inside a ZIP), so it never
        overrides a folder structure someone already curated.
        """
        existing_hashes = self.db.get_content_hashes()
        summary = IngestionSummary()

        resolved_paths: List[Path] = []
        flat_root_paths: Set[str] = set()  # extracted members with no internal zip subfolder
        for raw_path in file_paths:
            path = Path(raw_path.strip().strip('"\''))
            if path.suffix.lower() == ".zip":
                extracted, flat_roots = self._extract_zip(path, summary)  # already resolved paths
                resolved_paths.extend(extracted)
                flat_root_paths.update(flat_roots)
            elif path.exists():
                resolved_paths.append(path.resolve())
            else:
                resolved_paths.append(path)  # let _process_document report "file not found"

        meaningful_parents = {
            p.parent.name for p in resolved_paths
            if str(p) not in flat_root_paths
            and p.parent.name
            and p.parent.name.lower() not in GENERIC_FOLDER_NAMES
        }
        use_auto_partition = len(meaningful_parents) <= 1

        partition_map: Dict[str, str] = {}
        preloaded_texts: Dict[str, str] = {}
        if use_auto_partition and resolved_paths:
            partition_map, preloaded_texts = self._auto_partition_documents(
                [str(p) for p in resolved_paths]
            )

        for path in resolved_paths:
            self._process_document(
                path, existing_hashes, force, summary, partition_map,
                preloaded_text=preloaded_texts.get(str(path)),
            )
        if summary.processed:
            # entity_frequency (and Postgres term_df) must reflect the new
            # chunks, or high-fanout filtering silently no-ops on this corpus.
            try:
                self.db.finalize_ingest()
            except Exception as exc:
                print(f"Warning: post-ingest statistics refresh failed: {exc}")
        return summary

    def process_documents(self, file_paths: Sequence[str], force: bool = False) -> int:
        summary = self.process_uploads(file_paths, force=force)
        print(summary.as_markdown())
        return summary.processed


def select_files_dialog() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Select research papers",
            filetypes=[("Supported research files", "*.pdf *.txt *.md *.py *.js *.ipynb *.zip"), ("All files", "*.*")],
        )
        root.destroy()
        return list(files)
    except Exception as exc:
        print(f"File picker unavailable: {exc}")
        return []
