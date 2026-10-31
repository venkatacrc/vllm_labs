#!/usr/bin/env bash
# Lab 00 - Environment setup.
#   1. Create a dedicated conda env and install SGLang into it.
#   2. Pre-download every model used by the curriculum.
#   3. Boot a tiny smoke-test server to confirm everything works end to end.
#
# Safe to re-run: each step is idempotent (skips work that's already done).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/install_sglang.sh"
bash "$SCRIPT_DIR/download_models.sh"
bash "$SCRIPT_DIR/verify_install.sh"

echo
echo "Lab 00 complete. You're ready for lab01_fp32_baseline."
