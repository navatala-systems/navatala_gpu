#!/usr/bin/env bash
# Install the repo-local git hooks that enforce the Regen-Manifest-Trailer policy.
# Re-run safely; this script overwrites any existing hook files of the same name.
# Local hooks are best-effort — CI (.github/workflows/regen-trailer.yml) is the
# authoritative gate. See CONTRIBUTING.md for the policy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$(git -C "$ROOT" rev-parse --git-path hooks)"
CHECK="${ROOT}/tools/check_regen_trailer.sh"

if [[ ! -x "$CHECK" ]]; then
    chmod +x "$CHECK"
fi

mkdir -p "$HOOKS_DIR"

cat > "${HOOKS_DIR}/pre-commit" <<EOF
#!/usr/bin/env bash
exec "${CHECK}" --pre-commit
EOF
chmod +x "${HOOKS_DIR}/pre-commit"

cat > "${HOOKS_DIR}/commit-msg" <<EOF
#!/usr/bin/env bash
exec "${CHECK}" --commit-msg "\$1"
EOF
chmod +x "${HOOKS_DIR}/commit-msg"

echo "regen-trailer hooks installed in ${HOOKS_DIR}"
