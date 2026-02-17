# type: ignore

import os
import sys


def read_version() -> str:
    # Python 3.11+: tomllib is stdlib
    import tomllib

    with open("pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def main() -> None:
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if not tag.startswith("v"):
        print(f"Not a v* tag: {tag!r}")
        sys.exit(1)

    tag_ver = tag[1:]
    pkg_ver = read_version()

    print(f"Tag version: {tag_ver}")
    print(f"Package version: {pkg_ver}")

    if tag_ver != pkg_ver:
        print(f"ERROR: Tag v{tag_ver} does not match package version {pkg_ver}")
        sys.exit(1)


if __name__ == "__main__":
    main()
