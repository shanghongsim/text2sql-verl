#!/usr/bin/env bash
# Apply local patches to the installed verl (idempotent). Re-run after any reinstall of verl.
# Each patch is a backport/fix to upstream verl; drop it once a verl release contains the fix. See README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SP="$("$ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
VER="$("$ROOT/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("verl"))')"
[ "$VER" = "0.9.1" ] || { echo "patches target verl 0.9.1, found $VER — review before applying"; exit 1; }
for p in "$ROOT"/patches/verl-0.9.1-*.patch; do
  if patch -d "$SP" -p1 --dry-run -R -s -f < "$p" >/dev/null 2>&1; then
    echo "already applied: $(basename "$p")"
  else
    patch -d "$SP" -p1 -s < "$p" && echo "applied: $(basename "$p")"
  fi
done
