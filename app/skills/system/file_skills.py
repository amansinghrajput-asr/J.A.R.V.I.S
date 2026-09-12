"""File & Folder Skills for J.A.R.V.I.S. Phase 22.3.

Provides secure local filesystem operations (create folder, create file, read file,
list directory, rename, move, copy, delete, and open in explorer) with canonical
path containment validation, confirmation guardrails, symlink safety, and bounded limits.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, Final, List, Optional, Tuple, Union

from app.core.config import Settings
from app.core.container import ServiceContainer
from app.core.event_bus import EventBus
from app.core.logger import get_logger
from app.skills.base import SkillExecutionError
from app.skills.system.base_system_skill import BaseSystemSkill, SystemSkillResult
from app.skills.system.security import (
    ConfirmationRequiredError,
    SecurityPolicyViolationError,
    SystemConfirmationManager,
    SystemSecurityPolicy,
    validate_path,
)

logger = get_logger("SYSTEM.FILE_SKILLS")

# Safety operational bounds
DEFAULT_MAX_READ_BYTES: Final[int] = 5 * 1024 * 1024  # 5 MB
DEFAULT_MAX_DIR_ENTRIES: Final[int] = 100
MAX_ALLOWED_DIR_ENTRIES: Final[int] = 1000
DEFAULT_MAX_RECURSIVE_ITEMS: Final[int] = 500
DEFAULT_MAX_RECURSIVE_DEPTH: Final[int] = 5

# Intent regex matchers
_RE_CREATE_FOLDER = re.compile(
    r"^(?:create|make|new)\s+(?:folder|directory|dir)\s+(.+)$", re.IGNORECASE
)
_RE_CREATE_FILE = re.compile(r"^(?:create|make|write|new)\s+file\s+(.+)$", re.IGNORECASE)
_RE_READ_FILE = re.compile(
    r"^(?:read|view|show|cat|get\s+content)\s+(?:file\s+)?(.+)$", re.IGNORECASE
)
_RE_LIST_DIR = re.compile(
    r"^(?:list|ls|dir)\s+(?:files\s+in\s+|folder\s+|directory\s+)?(.+)$", re.IGNORECASE
)
_RE_RENAME = re.compile(r"^rename\s+(.+?)\s+(?:to|as)\s+(.+)$", re.IGNORECASE)
_RE_MOVE = re.compile(r"^move\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)
_RE_COPY = re.compile(r"^copy\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)
_RE_DELETE = re.compile(
    r"^(?:delete|remove|rm)\s+(?:file\s+|folder\s+|directory\s+|path\s+)?(.+)$",
    re.IGNORECASE,
)
_RE_EXPLORER = re.compile(r"^(?:open|show)\s+(.+?)\s+in\s+explorer$", re.IGNORECASE)


class FileSkills(BaseSystemSkill):
    """Production File and Folder Management Skills for J.A.R.V.I.S.

    Handles:
    - create_folder: Create directory tree safely
    - create_file: Write new file without silent overwrite
    - read_file: Read file with size limits and encoding safety
    - list_directory: Non-recursive, bounded directory enumeration
    - rename_path: Rename path without silent overwrite
    - move_path: Move path across locations with confirmation
    - copy_path: Copy files or directories with bounds
    - delete_path: Delete file or directory with mandatory confirmation
    - open_in_explorer: Reveal file or directory in Windows File Explorer
    """

    name: str = "file"
    description: str = (
        "Local filesystem and directory management with path security, "
        "confirmation guardrails, and bounded operations."
    )
    priority: int = 60
    tags: list[str] = ["file", "folder", "directory", "filesystem", "system", "windows"]
    permissions: set[str] = {"system:read", "system:write", "system:delete"}

    def __init__(
        self,
        *,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_dir_entries: int = DEFAULT_MAX_DIR_ENTRIES,
        max_recursive_items: int = DEFAULT_MAX_RECURSIVE_ITEMS,
        max_recursive_depth: int = DEFAULT_MAX_RECURSIVE_DEPTH,
        security_policy: Optional[SystemSecurityPolicy] = None,
        confirmation_manager: Optional[SystemConfirmationManager] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
        container: Optional[ServiceContainer] = None,
        event_bus: Optional[Union[EventBus, Any]] = None,
    ) -> None:
        """Initialize FileSkills instance."""
        super().__init__(
            name=self.name,
            description=self.description,
            priority=self.priority,
            tags=self.tags,
            permissions=self.permissions,
            security_policy=security_policy,
            confirmation_manager=confirmation_manager,
            config=config,
            logger=logger or get_logger("SKILL.FILE"),
            container=container,
            event_bus=event_bus,
        )
        self.max_read_bytes = max(1024, int(max_read_bytes))
        self.max_dir_entries = min(MAX_ALLOWED_DIR_ENTRIES, max(1, int(max_dir_entries)))
        self.max_recursive_items = max(1, int(max_recursive_items))
        self.max_recursive_depth = max(1, int(max_recursive_depth))

    def can_handle(self, command: Any) -> bool:
        """Evaluate whether this skill can handle the given command."""
        op, _, _, _ = self.parse_command(command)

        if op in (
            "create_folder",
            "create_file",
            "read_file",
            "list_directory",
            "rename_path",
            "move_path",
            "copy_path",
            "delete_path",
            "delete_file",
            "delete_folder",
            "open_in_explorer",
        ):
            return True

        if isinstance(command, str):
            clean = command.strip().lower()
            if _RE_CREATE_FOLDER.match(clean):
                return True
            if _RE_CREATE_FILE.match(clean):
                return True
            if _RE_READ_FILE.match(clean):
                return True
            if _RE_LIST_DIR.match(clean):
                return True
            if _RE_RENAME.match(clean):
                return True
            if _RE_MOVE.match(clean):
                return True
            if _RE_COPY.match(clean):
                return True
            if _RE_DELETE.match(clean):
                return True
            if _RE_EXPLORER.match(clean):
                return True

        return False

    def parse_command(
        self, command: Any
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """Normalize arbitrary command into structured components."""
        if isinstance(command, dict):
            return super().parse_command(command)

        text = str(command or "").strip()

        m_folder = _RE_CREATE_FOLDER.match(text)
        if m_folder:
            return "create_folder", m_folder.group(1).strip(), {}, None

        m_cfile = _RE_CREATE_FILE.match(text)
        if m_cfile:
            return "create_file", m_cfile.group(1).strip(), {}, None

        m_read = _RE_READ_FILE.match(text)
        if m_read:
            return "read_file", m_read.group(1).strip(), {}, None

        m_list = _RE_LIST_DIR.match(text)
        if m_list:
            return "list_directory", m_list.group(1).strip(), {}, None

        m_ren = _RE_RENAME.match(text)
        if m_ren:
            return "rename_path", m_ren.group(1).strip(), {"destination": m_ren.group(2).strip()}, None

        m_mv = _RE_MOVE.match(text)
        if m_mv:
            return "move_path", m_mv.group(1).strip(), {"destination": m_mv.group(2).strip()}, None

        m_cp = _RE_COPY.match(text)
        if m_cp:
            return "copy_path", m_cp.group(1).strip(), {"destination": m_cp.group(2).strip()}, None

        m_del = _RE_DELETE.match(text)
        if m_del:
            return "delete_path", m_del.group(1).strip(), {}, None

        m_exp = _RE_EXPLORER.match(text)
        if m_exp:
            return "open_in_explorer", m_exp.group(1).strip(), {}, None

        return super().parse_command(command)

    def _execute_operation(
        self, operation: str, target: Optional[str], parameters: Dict[str, Any]
    ) -> Any:
        """Internal operation dispatcher for FileSkills."""
        op = operation.strip().lower()

        if op in ("create_folder", "create_dir", "mkdir"):
            return self.create_folder(target, parameters)

        if op in ("create_file", "write_file"):
            return self.create_file(target, parameters)

        if op in ("read_file", "view_file", "cat"):
            return self.read_file(target, parameters)

        if op in ("list_directory", "list_dir", "ls", "dir"):
            return self.list_directory(target, parameters)

        if op in ("rename_path", "rename"):
            return self.rename_path(target, parameters)

        if op in ("move_path", "move", "mv"):
            return self.move_path(target, parameters)

        if op in ("copy_path", "copy", "cp"):
            return self.copy_path(target, parameters)

        if op in ("delete_path", "delete_file", "delete_folder", "remove_file", "remove_directory", "rm"):
            return self.delete_path(target, parameters)

        if op in ("open_in_explorer", "reveal_in_explorer", "explorer"):
            return self.open_in_explorer(target, parameters)

        raise NotImplementedError(f"Operation '{op}' is not supported by {self.name}.")

    def _resolve_validated_path(self, path_str: Optional[str]) -> Path:
        """Helper to validate path against security policy and return canonical Path."""
        if not path_str or not str(path_str).strip():
            raise SkillExecutionError("Filesystem operation requires a non-empty path.")

        allowed_roots = getattr(self.security_policy, "_allowed_roots", None)
        try:
            return validate_path(path_str, allowed_roots=allowed_roots, allow_system_dirs=False)
        except SecurityPolicyViolationError as exc:
            raise SkillExecutionError(f"Security policy violation for path '{path_str}': {exc}") from exc

    def create_folder(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely create a new directory tree."""
        p = self._resolve_validated_path(target)

        if p.exists():
            if p.is_dir():
                return {"path": str(p), "created": False, "exists": True}
            raise SkillExecutionError(f"Path '{p}' exists and is a file, not a directory.")

        try:
            p.mkdir(parents=True, exist_ok=True)
            self.logger.info("Created directory: '%s'", p)
            return {"path": str(p), "created": True}
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied creating directory '{p}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to create directory '{p}': {exc}") from exc

    def create_file(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely create a new file, preventing silent overwrite without confirmation."""
        p = self._resolve_validated_path(target)
        content = parameters.get("content", "")
        overwrite = bool(parameters.get("overwrite", False))

        if p.exists() and not overwrite:
            raise SkillExecutionError(
                f"File '{p}' already exists. Overwrite requires explicit confirmation."
            )

        if p.is_dir():
            raise SkillExecutionError(f"Target '{p}' is an existing directory.")

        try:
            # Ensure parent directories exist
            p.parent.mkdir(parents=True, exist_ok=True)

            text_content = str(content)
            p.write_text(text_content, encoding="utf-8", errors="replace")
            bytes_written = len(text_content.encode("utf-8"))
            self.logger.info("Created file '%s' (%d bytes written)", p, bytes_written)

            return {
                "path": str(p),
                "created": True,
                "bytes_written": bytes_written,
                "overwritten": p.exists() and overwrite,
            }
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied writing file '{p}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to create file '{p}': {exc}") from exc

    def read_file(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely read file content with bounded size limits and encoding fallbacks."""
        p = self._resolve_validated_path(target)
        max_bytes = int(parameters.get("max_bytes") or self.max_read_bytes)

        if not p.exists():
            raise SkillExecutionError(f"File '{p}' does not exist.")

        if p.is_dir():
            raise SkillExecutionError(f"Target '{p}' is a directory, not a regular file.")

        try:
            file_size = p.stat().st_size
            if file_size > max_bytes:
                raise SkillExecutionError(
                    f"File size ({file_size} bytes) exceeds maximum permitted read size ({max_bytes} bytes)."
                )

            # Read with UTF-8, fallback to Latin-1
            raw_bytes = p.read_bytes()
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("latin-1", errors="replace")

            return {
                "path": str(p),
                "size_bytes": file_size,
                "content": content,
            }
        except SkillExecutionError:
            raise
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied reading file '{p}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to read file '{p}': {exc}") from exc

    def list_directory(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely enumerate entries in a directory with bounded limits."""
        p = self._resolve_validated_path(target or ".")
        limit = min(
            MAX_ALLOWED_DIR_ENTRIES,
            int(parameters.get("limit") or self.max_dir_entries),
        )
        pattern = str(parameters.get("pattern") or "*")

        if not p.exists():
            raise SkillExecutionError(f"Directory '{p}' does not exist.")

        if not p.is_dir():
            raise SkillExecutionError(f"Target '{p}' is a file, not a directory.")

        entries: List[Dict[str, Any]] = []
        has_more = False

        try:
            matched_items = list(p.glob(pattern))
            if len(matched_items) > limit:
                has_more = True
                matched_items = matched_items[:limit]

            for item in matched_items:
                is_dir = item.is_dir()
                is_symlink = item.is_symlink() or os.path.islink(item)
                size_bytes = 0
                if not is_dir and not is_symlink:
                    try:
                        size_bytes = item.stat().st_size
                    except Exception:
                        size_bytes = 0

                entries.append({
                    "name": item.name,
                    "is_dir": is_dir,
                    "is_symlink": is_symlink,
                    "size_bytes": size_bytes,
                })

            return {
                "path": str(p),
                "entries": entries,
                "total_returned": len(entries),
                "has_more": has_more,
            }
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied accessing directory '{p}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to list directory '{p}': {exc}") from exc

    def rename_path(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely rename a file or directory."""
        src = self._resolve_validated_path(target)
        dest_raw = parameters.get("destination") or parameters.get("target")
        dest = self._resolve_validated_path(str(dest_raw or ""))
        overwrite = bool(parameters.get("overwrite", False))

        if not src.exists():
            raise SkillExecutionError(f"Source '{src}' does not exist.")

        if dest.exists() and not overwrite:
            raise SkillExecutionError(
                f"Destination '{dest}' already exists. Set overwrite=True with confirmation to replace."
            )

        try:
            if dest.exists() and overwrite:
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

            src.rename(dest)
            self.logger.info("Renamed '%s' to '%s'", src, dest)
            return {
                "source": str(src),
                "destination": str(dest),
                "renamed": True,
            }
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied renaming '{src}' to '{dest}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to rename '{src}' to '{dest}': {exc}") from exc

    def move_path(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely move a file or directory across paths or roots."""
        src = self._resolve_validated_path(target)
        dest_raw = parameters.get("destination") or parameters.get("target")
        dest = self._resolve_validated_path(str(dest_raw or ""))
        overwrite = bool(parameters.get("overwrite", False))

        if not src.exists():
            raise SkillExecutionError(f"Source '{src}' does not exist.")

        if dest.exists() and not overwrite:
            raise SkillExecutionError(
                f"Destination '{dest}' already exists. Overwrite requires confirmation."
            )

        try:
            if dest.exists() and overwrite:
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

            shutil.move(str(src), str(dest))
            self.logger.info("Moved '%s' to '%s'", src, dest)
            return {
                "source": str(src),
                "destination": str(dest),
                "moved": True,
            }
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied moving '{src}' to '{dest}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to move '{src}' to '{dest}': {exc}") from exc

    def copy_path(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely copy a file or directory."""
        src = self._resolve_validated_path(target)
        dest_raw = parameters.get("destination") or parameters.get("target")
        dest = self._resolve_validated_path(str(dest_raw or ""))
        overwrite = bool(parameters.get("overwrite", False))

        if not src.exists():
            raise SkillExecutionError(f"Source '{src}' does not exist.")

        if dest.exists() and not overwrite:
            raise SkillExecutionError(
                f"Destination '{dest}' already exists. Overwrite requires confirmation."
            )

        try:
            if src.is_dir():
                # Directory copy: symlinks=False to avoid copying target of dangerous links
                if dest.exists() and overwrite:
                    shutil.rmtree(dest)
                shutil.copytree(str(src), str(dest), symlinks=False)
            else:
                # File copy: follow_symlinks=False
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dest), follow_symlinks=False)

            self.logger.info("Copied '%s' to '%s'", src, dest)
            return {
                "source": str(src),
                "destination": str(dest),
                "copied": True,
            }
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied copying '{src}' to '{dest}': {exc}") from exc
        except Exception as exc:
            raise SkillExecutionError(f"Failed to copy '{src}' to '{dest}': {exc}") from exc

    def delete_path(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely delete a file or directory with symlink safety and recursion limits.

        Strictly enforces confirmation via BaseSystemSkill and SystemSecurityPolicy.
        """
        p = self._resolve_validated_path(target)
        recursive = bool(parameters.get("recursive", False))

        # Block root deletion
        if p.anchor == str(p) or len(p.parts) <= 1:
            raise SecurityPolicyViolationError(
                f"Deletion of filesystem root '{p}' is strictly prohibited."
            )

        if not p.exists() and not p.is_symlink() and not os.path.islink(p):
            raise SkillExecutionError(f"Path '{p}' does not exist.")

        try:
            # 1. Handle symlinks / junctions: unlink link itself without following into target
            if p.is_symlink() or os.path.islink(p):
                if p.is_dir() and sys.platform == "win32":
                    os.rmdir(p)
                else:
                    os.unlink(p)
                self.logger.info("Unlinked symlink/junction: '%s'", p)
                return {"path": str(p), "deleted": True, "type": "symlink"}

            # 2. Handle regular file
            if p.is_file():
                p.unlink()
                self.logger.info("Deleted file: '%s'", p)
                return {"path": str(p), "deleted": True, "type": "file"}

            # 3. Handle directory
            if p.is_dir():
                if not recursive:
                    # Non-recursive: will fail if directory is not empty
                    p.rmdir()
                    self.logger.info("Deleted empty directory: '%s'", p)
                    return {"path": str(p), "deleted": True, "type": "directory"}

                # Recursive deletion: verify recursion item count and depth bounds
                item_count = 0
                for root, dirs, files in os.walk(p):
                    rel_depth = len(Path(root).relative_to(p).parts)
                    if rel_depth > self.max_recursive_depth:
                        raise SkillExecutionError(
                            f"Directory depth ({rel_depth}) exceeds maximum recursive deletion limit ({self.max_recursive_depth})."
                        )
                    item_count += len(dirs) + len(files)
                    if item_count > self.max_recursive_items:
                        raise SkillExecutionError(
                            f"Item count ({item_count}) exceeds maximum recursive deletion limit ({self.max_recursive_items})."
                        )

                shutil.rmtree(p)
                self.logger.info("Deleted directory recursively: '%s' (%d items)", p, item_count)
                return {"path": str(p), "deleted": True, "type": "directory", "items_deleted": item_count}

            raise SkillExecutionError(f"Unsupported filesystem object type at '{p}'.")
        except SkillExecutionError:
            raise
        except PermissionError as exc:
            raise SkillExecutionError(f"Permission denied deleting '{p}': {exc}") from exc
        except OSError as exc:
            raise SkillExecutionError(f"Filesystem error deleting '{p}': {exc}") from exc

    def open_in_explorer(self, target: Optional[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely reveal a file or directory in Windows File Explorer."""
        p = self._resolve_validated_path(target or ".")

        if not p.exists():
            raise SkillExecutionError(f"Path '{p}' does not exist.")

        try:
            if sys.platform == "win32":
                if p.is_file():
                    subprocess.Popen(
                        ["explorer.exe", f"/select,{str(p)}"],
                        shell=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    subprocess.Popen(
                        ["explorer.exe", str(p)],
                        shell=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            self.logger.info("Opened path in explorer: '%s'", p)
            return {"path": str(p), "opened": True}
        except Exception as exc:
            raise SkillExecutionError(f"Failed to open explorer for '{p}': {exc}") from exc


__all__ = [
    "DEFAULT_MAX_DIR_ENTRIES",
    "DEFAULT_MAX_READ_BYTES",
    "DEFAULT_MAX_RECURSIVE_DEPTH",
    "DEFAULT_MAX_RECURSIVE_ITEMS",
    "FileSkills",
    "MAX_ALLOWED_DIR_ENTRIES",
]
