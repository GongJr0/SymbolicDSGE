from dotenv import load_dotenv, find_dotenv
import os
import subprocess
import shutil
import sys


def main() -> None:
    load_dotenv(find_dotenv())
    api = os.getenv("PYPI_API")
    build_path = "./dist"
    package_name = "SymbolicDSGE"

    if api is None:
        print("PYPI_API environment variable not set. Exiting.")
        sys.exit(1)

    print("Removing previous builds...")
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    print("Building the package...")
    build_result = subprocess.run(["uv", "build"], stderr=subprocess.PIPE)
    build_err = build_result.returncode
    err_out = build_result.stderr.decode()

    if not build_err:  # not 0 == True
        print("Build Successful!")
    else:
        print(
            f"Build failed with error code {build_err}."
            f" Error output:\n{err_out}\nExiting."
        )
        sys.exit(1)

    print("Publishing the package to PyPI...")

    env = {**os.environ, "TWINE_USERNAME": "__token__", "TWINE_PASSWORD": api}

    publish_result = subprocess.run(
        [
            "uv",
            "publish",
        ],
        stderr=subprocess.PIPE,
        env=env,
    )
    publish_err = publish_result.returncode
    err_out = publish_result.stderr.decode()
    if not publish_err:
        print("Publish Successful!")
    else:
        print(
            f"Publish failed with error code {publish_err}."
            f" Error output:\n{err_out}\nExiting."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
