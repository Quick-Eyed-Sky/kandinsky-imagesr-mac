#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BUILD="$ROOT/build"
APP="$BUILD/Kandinsky ImageSR.app"
RES="$APP/Contents/Resources"
PYTHON_URL='https://github.com/astral-sh/python-build-standalone/releases/download/20261124/cpython-3.11.18%2B20261124-aarch64-apple-darwin-install_only_stripped.tar.gz'
PYTHON_HASH='e9840948169972436f80241773458cdb95923a639f652c01bdc3f1066e45f113'
mkdir -p "$RES/runtime" "$APP/Contents/MacOS" "$BUILD/downloads"
curl -fL --retry 3 "$PYTHON_URL" -o "$BUILD/downloads/python.tar.gz"
printf '%s  %s\n' "$PYTHON_HASH" "$BUILD/downloads/python.tar.gz" | shasum -a 256 -c -
tar -xzf "$BUILD/downloads/python.tar.gz" -C "$RES/runtime" --strip-components=1
PY="$RES/runtime/bin/python3"
"$PY" -m pip install --no-compile mlx==0.32.5 numpy==2.4.6 pillow==13.0.0 pillow-heif==1.9.0
"$PY" scripts/download_model.py "$BUILD/downloads/model"
"$PY" Source/pack_model.py "$BUILD/downloads/model" "$RES/model"
cp Source/worker.py Source/metal_model.py "$RES/"
cp licenses/Kandinsky-MIT.txt "$RES/model/LICENSE-Kandinsky"
cp licenses/Apache-2.0.txt "$RES/model/LICENSE-Apache-2.0"
cp NOTICE.txt "$RES/model/NOTICE.txt"
cp Info.plist "$APP/Contents/Info.plist"
"$PY" Source/icon.py "$BUILD/AppIcon.iconset"
iconutil -c icns "$BUILD/AppIcon.iconset" -o "$RES/AppIcon.icns"
if [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
xcrun swiftc -target arm64-apple-macos26.0 -O -o "$APP/Contents/MacOS/Kandinsky" Source/Studio.swift Source/Window.swift
codesign --force --sign - "$APP"
echo "Built $APP"
