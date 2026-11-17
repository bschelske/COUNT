# PyInstaller build file for the COUNT app. Build with:
#     pip install -r requirements.txt pyinstaller
#     pyinstaller COUNT.spec
# The app is made in dist/: dist/COUNT/COUNT.exe on Windows, dist/COUNT.app on macOS.
# GitHub Actions builds it for Windows and macOS (.github/workflows/build-app.yml). See DEVELOPERS.md.
import os
import platform
import sys

from PyInstaller.utils.hooks import get_package_paths

datas = [("COUNT/settings.json", "COUNT"), ("COUNT/count.ico", "COUNT")]
binaries = []

# The Nikon SDK used to read .nd2 files is a set of libraries inside pims_nd2, which pims only imports when an
# .nd2 file is opened, so PyInstaller doesn't find it by itself. Only this system's libraries are collected.
hiddenimports = ["pims_nd2", "pims_nd2.ND2SDK", "pims_nd2.nd2reader"]
_, pims_nd2_dir = get_package_paths("pims_nd2")
sdk_dir = os.path.join(pims_nd2_dir, "ND2SDK")
if sys.platform == "win32":
    sdk_libraries = "win/x64"
elif sys.platform == "darwin":
    # Intel-only. The framework's top-level files are broken symlinks in the pims_nd2 package (and confuse
    # PyInstaller), so only the real libraries in Versions/1 are collected, which is all pims_nd2 loads.
    sdk_libraries = "osx/nd2sdk.framework/Versions/1" if platform.machine() == "x86_64" else None
else:
    sdk_libraries = "linux"
if sdk_libraries:
    folder = os.path.join(sdk_dir, *sdk_libraries.split("/"))
    for name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, name)) and not name.endswith(".h"):
            binaries.append((os.path.join(folder, name), "/".join(["pims_nd2", "ND2SDK", sdk_libraries])))

a = Analysis(
    ["COUNT/main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        # Only used by bg_plotting.py and box.py, which aren't part of the app
        "pandas", "scipy", "boxsdk", "dotenv",
        # Not used, but some libraries can optionally use them
        "IPython", "jupyter_client", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="COUNT",
    console=False,  # no console window: progress is shown in COUNT's own progress window
    icon="COUNT/count.ico",
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="COUNT", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="COUNT.app",
        icon="COUNT/count.ico",
        bundle_identifier="io.github.bschelske.count",
        info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "1.0"},
    )
