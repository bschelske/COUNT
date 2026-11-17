"""
Where COUNT keeps its files, both when run from the code (python COUNT/main.py) and as the app built with
PyInstaller (see DEVELOPERS.md).

Run from the code, everything stays as it always was: the default settings are COUNT/settings.json and a
relative results folder is relative to where COUNT is run from.

In the app, the files next to the code are inside the app and are replaced by every new version (and on a Mac
can't be written to), so:
- saved default settings go in the user's settings folder (see user_data_dir), and the settings.json that came
  with the app is used until the user saves their own;
- a relative results folder such as "results/" goes in Documents/COUNT, not wherever the app happened to start.
"""
import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))  # True in the app built with PyInstaller

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_SETTINGS_PATH = os.path.join(PACKAGE_DIR, "settings.json")
ICON_PATH = os.path.join(PACKAGE_DIR, "count.ico")


def user_data_dir() -> str:
    """The folder for the app's own files: %APPDATA%\\COUNT, ~/Library/Application Support/COUNT or ~/.config/COUNT."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "COUNT")


def default_settings_path() -> str:
    """Where "Save as default settings" saves, and the settings file read when COUNT starts."""
    return os.path.join(user_data_dir(), "settings.json") if FROZEN else BUNDLED_SETTINGS_PATH


def documents_dir() -> str:
    """The user's Documents folder, or their home folder if there isn't one."""
    documents = os.path.join(os.path.expanduser("~"), "Documents")
    return documents if os.path.isdir(documents) else os.path.expanduser("~")


def results_path(path: str) -> str:
    """The results folder to use for `path` from the settings. In the app, relative paths go in Documents/COUNT."""
    path = os.path.expanduser(path)
    if FROZEN and not os.path.isabs(path):
        return os.path.normpath(os.path.join(documents_dir(), "COUNT", path))
    return path
