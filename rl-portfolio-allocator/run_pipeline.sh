#!/usr/bin/env bash
# Keep the package-local entry point in sync with the repository pipeline.
set -Eeuo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_pipeline.sh" "$@"
