#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/output"
BUILD_OUT="$ROOT/output/.build"
mkdir -p "$BUILD_OUT"
cd "$ROOT/src"
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_OUT" v3.0.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_OUT" v3.0.tex
cp "$BUILD_OUT/v3.0.pdf" "$ROOT/output/v3.0.pdf"
cp "$BUILD_OUT/v3.0.log" "$ROOT/output/v3.0.log"
cp "$BUILD_OUT/v3.0.pdf" "$ROOT/v3.0.pdf"
cp "$BUILD_OUT/v3.0.pdf" "$ROOT/../v3.0.pdf"
echo "PDF written to $ROOT/output/v3.0.pdf"
echo "Release copy written to $ROOT/v3.0.pdf"
echo "Canonical copy written to $ROOT/../v3.0.pdf"
