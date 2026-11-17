# PyInstaller build file for the COUNT app. Build with:
#     pip install -r requirements.txt pyinstaller
#     pyinstaller COUNT.spec
# The app is made in dist/: dist/COUNT/COUNT.exe on Windows, dist/COUNT.app on macOS.
# GitHub Actions builds it for Windows and macOS (.github/workflows/build-app.yml). See DEVELOPERS.md.
import sys

from PyInstaller.utils.hooks import collect_all

datas = [("COUNT/settings.json", "COUNT"), ("COUNT/count.ico", "COUNT")]
binaries = []
hiddenimports = []

# The Nikon SDK used to read .nd2 files is a set of libraries inside pims_nd2, which pims only imports when an
# .nd2 file is opened, so PyInstaller can't find it by itself
for package in ("pims_nd2",):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports + ["pims_nd2.ND2SDK"]

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
