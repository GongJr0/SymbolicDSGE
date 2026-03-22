#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import subprocess


def find_env_file(path_arg: str | None) -> Path | None:
    if path_arg:
        p = Path(path_arg)
        return p if p.exists() else None

    p = Path(".env")
    return p if p.exists() else None


def find_all_env_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", ":(glob)*.env"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    tracked = result.stdout.strip().splitlines()
    print(f"Found tracked .env files: {tracked}")
    return [Path(f) for f in tracked]


def file_is_empty(path: Path) -> bool:
    return path.stat().st_size == 0


def is_skip_worktree(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "-v", str(path)], capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout:
        return False
    return result.stdout.startswith("S")


def staged_env_is_nonempty(path: Path) -> bool:
    result = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=str, default=None)
    args = parser.parse_args()

    # --- Global mode: check all tracked .env files ---
    if args.env_file is None:
        for f in find_all_env_files():
            if (not is_skip_worktree(f)) and (not file_is_empty(f)):
                raise SystemExit(
                    f"{f} contains file contents.\n"
                    "Clear it and commit the empty file.\n"
                    "Then freeze it with:\n"
                    "  git update-index --skip-worktree <path-to-.env>"
                )
            if staged_env_is_nonempty(f):
                raise SystemExit(
                    f"Staged {f} contains file contents.\n"
                    "Clear it and stage the empty version instead."
                )
        sys.exit(0)

    # --- Single-file mode ---
    env_path = find_env_file(args.env_file)

    if env_path is None:
        sys.exit(0)

    # Skip-worktree means the file is intentionally local-only
    if is_skip_worktree(env_path):
        sys.exit(0)

    # Working tree contains secrets → block
    if not file_is_empty(env_path):
        raise SystemExit(
            ".env contains file contents.\n"
            "Clear it and commit the empty file.\n"
            "Then freeze it with:\n"
            "  git update-index --skip-worktree <path-to-.env>"
        )

    # Staged version contains secrets → block
    if staged_env_is_nonempty(env_path):
        raise SystemExit(
            "Staged .env contains file contents.\n"
            "Clear it and stage the empty version instead."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
