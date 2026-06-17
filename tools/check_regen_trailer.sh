#!/usr/bin/env bash
# Enforce the Regen-Manifest-Trailer policy for protected generated paths.
# Local invocations are best-effort (git commit --no-verify bypasses them);
# the CI workflow under .github/workflows/regen-trailer.yml is the authoritative
# gate. See CONTRIBUTING.md ("Regeneration trailer policy for generated paths").
set -euo pipefail

PROTECTED_REGEX='^(kernels/|python/navatala_gpu/|bindings/)'
TRAILER_KEY='Regen-Manifest-Trailer'

current_manifest_id() {
    local meta="python/release_metadata.json"
    if [[ ! -f "$meta" ]]; then
        echo "regen-trailer-check: $meta not found; cannot validate trailer." >&2
        return 1
    fi
    python3 -c "import json,sys;print(json.load(open('$meta')).get('manifestId',''))"
}

paths_in_change() {
    case "${1:-}" in
        --staged) git diff --cached --name-only ;;
        --range) shift; git diff --name-only "$1" ;;
        *) git diff --cached --name-only ;;
    esac
}

protected_paths_in_change() {
    paths_in_change "$@" | grep -E "$PROTECTED_REGEX" || true
}

usage() {
    cat <<'USAGE'
Usage: tools/check_regen_trailer.sh <mode> [args]

Modes:
  --pre-commit             Warn on staged edits to protected paths.
  --commit-msg <file>      Reject commit if message lacks a valid trailer
                           and any staged path is protected.
  --ci <base-ref>          Reject the diff between HEAD and <base-ref>
                           if any protected path is touched without a
                           valid trailer in the most recent commit message.
USAGE
}

mode="${1:-}"

case "$mode" in
  --pre-commit)
      hits="$(protected_paths_in_change --staged)"
      if [[ -n "$hits" ]]; then
          echo "regen-trailer-check: staged edits touch protected paths:" >&2
          echo "$hits" | sed 's/^/  /' >&2
          echo "regen-trailer-check: commit message must include a" >&2
          echo "  '${TRAILER_KEY}: <manifest-id>' trailer (see CONTRIBUTING.md)." >&2
      fi
      exit 0
      ;;
  --commit-msg)
      msg_file="${2:-}"
      if [[ -z "$msg_file" || ! -f "$msg_file" ]]; then
          echo "regen-trailer-check: --commit-msg requires a path to the message file." >&2
          exit 2
      fi
      hits="$(protected_paths_in_change --staged)"
      if [[ -z "$hits" ]]; then
          exit 0
      fi
      trailer_value="$(grep -E "^${TRAILER_KEY}:[[:space:]]*" "$msg_file" | head -1 | sed -E "s/^${TRAILER_KEY}:[[:space:]]*//")"
      if [[ -z "$trailer_value" ]]; then
          echo "regen-trailer-check: commit touches protected paths but the" >&2
          echo "  commit message has no '${TRAILER_KEY}:' trailer." >&2
          echo "$hits" | sed 's/^/  /' >&2
          exit 1
      fi
      expected="$(current_manifest_id || true)"
      if [[ -n "$expected" && "$trailer_value" != "$expected" ]]; then
          echo "regen-trailer-check: trailer manifest id '$trailer_value' does" >&2
          echo "  not match release_metadata.json id '$expected'." >&2
          exit 1
      fi
      exit 0
      ;;
  --ci)
      base="${2:-origin/main}"
      hits="$(protected_paths_in_change --range "$base")"
      if [[ -z "$hits" ]]; then
          exit 0
      fi
      trailer_value="$(git log -1 --format=%B HEAD | grep -E "^${TRAILER_KEY}:[[:space:]]*" | head -1 | sed -E "s/^${TRAILER_KEY}:[[:space:]]*//")"
      if [[ -z "$trailer_value" ]]; then
          echo "regen-trailer-check: HEAD touches protected paths but lacks a" >&2
          echo "  '${TRAILER_KEY}:' trailer in the commit message." >&2
          echo "$hits" | sed 's/^/  /' >&2
          exit 1
      fi
      expected="$(current_manifest_id || true)"
      if [[ -n "$expected" && "$trailer_value" != "$expected" ]]; then
          echo "regen-trailer-check: trailer manifest id '$trailer_value' does" >&2
          echo "  not match release_metadata.json id '$expected'." >&2
          exit 1
      fi
      exit 0
      ;;
  -h|--help|"")
      usage
      exit 0
      ;;
  *)
      echo "regen-trailer-check: unknown mode '$mode'." >&2
      usage >&2
      exit 2
      ;;
esac
