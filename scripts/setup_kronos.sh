#!/usr/bin/env bash
# Clone upstream Kronos at the pinned commit into vendor/Kronos.
set -euo pipefail
cd "$(dirname "$0")/.."
commit="$(cat vendor/KRONOS_COMMIT)"
if [ ! -d vendor/Kronos/.git ]; then
  git clone https://github.com/shiyu-coder/Kronos.git vendor/Kronos
fi
git -C vendor/Kronos checkout --quiet "$commit"
echo "Kronos at $(git -C vendor/Kronos rev-parse HEAD)"
