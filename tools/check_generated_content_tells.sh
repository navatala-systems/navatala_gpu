#!/usr/bin/env bash
# Aggregate acceptance check for #1513 generator-content cleanup.
# Run from the published navatala_gpu tree root. Exits non-zero on first
# failure so it can wire cleanly into CI.
set -euo pipefail

ROOT="${1:-.}"
if [[ ! -d "$ROOT" ]]; then
    echo "$0: root directory not found: $ROOT" >&2
    exit 2
fi
cd "$ROOT"

# ----------------------------------------------------------------------------
# Allowlist: files that may carry checker implementation text. Public
# manifests/docs must not carry private namespace or IR type strings.
# ----------------------------------------------------------------------------
ALLOWLIST=(
    "tools/check_generated_content_tells.sh"
    "tools/check_regen_trailer.sh"
    "tools/install_hooks.sh"
    "tools/check_release_hygiene.py"
    ".github/workflows/regen-trailer.yml"
    ".github/workflows/wheel-hygiene.yml"
    ".github/workflows/publish-pypi.yml"
)

is_allowlisted() {
    local f="$1"
    for entry in "${ALLOWLIST[@]}"; do
        [[ "$f" == "$entry" ]] && return 0
    done
    return 1
}

failures=0
fail() {
    echo "FAIL: $1" >&2
    failures=$((failures + 1))
}

pass() {
    echo "  PASS: $1"
}

echo "=== Path checks ==="

# Phase 2.7 + 3.7: no file paths containing generator-content tells
PATH_HITS=$(find . -type f \
    \( -iname '*generated*' -o -iname '*codegen*' -o -iname '*CodeGenGPU*' -o -iname '*Calculations_GPU*' \) \
    -not -path '*/.pytest_cache/*' -not -path '*/__pycache__/*' \
    -not -path '*/.git/*' \
    | sed 's|^\./||')
if [[ -n "$PATH_HITS" ]]; then
    unexpected=""
    while IFS= read -r f; do
        if ! is_allowlisted "$f"; then
            unexpected+="$f"$'\n'
        fi
    done <<< "$PATH_HITS"
    if [[ -n "$unexpected" ]]; then
        fail "Public-tree file paths contain generator-content tells:"
        printf '  %s\n' $unexpected >&2
    else
        pass "All path hits are on the provenance allowlist"
    fi
else
    pass "No file paths match generator-content patterns"
fi

# Phase 2.7 + 3.7: no directories named after generator-content patterns
DIR_HITS=$(find . -type d \
    \( -iname '*generated*' -o -iname '*codegen*' -o -iname '*CodeGenGPU*' -o -iname '*Calculations_GPU*' \) \
    -not -path '*/.git/*' \
    | sed 's|^\./||')
if [[ -n "$DIR_HITS" ]]; then
    fail "Public-tree directories contain generator-content tells:"
    printf '  %s\n' $DIR_HITS >&2
else
    pass "No directories match generator-content patterns"
fi

# Phase 3.7: no Python cache / pytest cache / egg-info
CACHE_HITS=$(find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.pytest_cache' -o -name '*.egg-info' \) \
    -not -path '*/.git/*' | sed 's|^\./||')
if [[ -n "$CACHE_HITS" ]]; then
    fail "Public tree contains Python/pytest cache or egg-info:"
    printf '  %s\n' $CACHE_HITS >&2
else
    pass "No Python cache or egg-info present"
fi

echo
echo "=== Content checks ==="

# Phase 1.9: no banner string in any Python module
BANNER_HITS=$(grep -rln "Generated package artifact" python 2>/dev/null \
    || true)
if [[ -n "$BANNER_HITS" ]]; then
    fail "Found 'Generated package artifact' banner in Python tree:"
    printf '  %s\n' $BANNER_HITS >&2
else
    pass "No 'Generated package artifact' banner in Python tree"
fi

# Phase 1.9: no '# Manifest-ID:' comment outside _meta.py
MID_HITS=$(grep -rln "Manifest-ID:" python --include='*.py' 2>/dev/null \
    | grep -v 'python/navatala_gpu/_meta.py' || true)
if [[ -n "$MID_HITS" ]]; then
    fail "Found '# Manifest-ID:' comment outside the _meta.py allowlist:"
    printf '  %s\n' $MID_HITS >&2
else
    pass "No '# Manifest-ID:' comments outside _meta.py"
fi

# Private namespace/type leak checks. The public package may expose stable
# public contract ids, but not the private Lean namespace or GPU IR type system.
PRIVATE_HITS=$(grep -RInE 'CodeGenGPU\.|GPU\.GPUType|Calculations_GPU|Contract: CodeGenGPU' . 2>/dev/null \
    | sed 's|^\./||' | grep -v 'tools/check_generated_content_tells.sh' || true)
if [[ -n "$PRIVATE_HITS" ]]; then
    found_unexpected=""
    while IFS= read -r f; do
        file="${f%%:*}"
        if ! is_allowlisted "$file"; then
            found_unexpected+="$f"$'\n'
        fi
    done <<< "$PRIVATE_HITS"
    if [[ -n "$found_unexpected" ]]; then
        fail "Found private namespace/type content outside allowlist:"
        printf '  %s\n' $found_unexpected >&2
    else
        pass "Private namespace/type content appears only inside checker implementation"
    fi
else
    pass "No private namespace/type content found"
fi

# Phase 3.7: no __CODEGEN_GPU__ raw-string delimiter
DELIM_HITS=$(grep -rln "__CODEGEN_GPU__" . 2>/dev/null \
    | sed 's|^\./||' | grep -v 'tools/check_generated_content_tells.sh' || true)
if [[ -n "$DELIM_HITS" ]]; then
    fail "Found '__CODEGEN_GPU__' raw-string delimiter:"
    printf '  %s\n' $DELIM_HITS >&2
else
    pass "No '__CODEGEN_GPU__' raw-string delimiter"
fi

# Phase 4.9: README/CONTRIBUTING/KERNELS don't lead with generator framing
PROSE_HITS=$(grep -ln 'mechanical translations\|generated artifacts\|upstream specification/compiler' \
    README.md CONTRIBUTING.md docs/KERNELS.md 2>/dev/null \
    | grep -v 'docs/KERNELS.md' || true)
# (docs/KERNELS.md may legitimately discuss generation in technical detail —
# we only enforce against the top-level README/CONTRIBUTING.)
if [[ -n "$PROSE_HITS" ]]; then
    fail "Generator-framing prose in user-facing docs:"
    printf '  %s\n' $PROSE_HITS >&2
else
    pass "No generator-framing prose in README/CONTRIBUTING"
fi

# Phase 4.9: no public doc file starts with 'This page is generated from'
DOCGEN_HITS=$(grep -l '^This page is generated from' docs/*.md 2>/dev/null || true)
if [[ -n "$DOCGEN_HITS" ]]; then
    fail "Public docs lead with 'This page is generated from':"
    printf '  %s\n' $DOCGEN_HITS >&2
else
    pass "No public doc starts with 'This page is generated from'"
fi

# Phase 2.7: avg C ABI kernel symbol length under 40 chars.
# Per #1513 §2.3, kernels live under kernels/<backend>/<module>/<stem>.<ext>
# instead of a flat kernels/<backend>/ listing — so the scan must recurse.
# Per #1513 §3 the published tree may ship without CUDA (alpha drops some
# backends), so scan every backend directory rather than only `kernels/cuda`.
# Uses `find -exec grep` (instead of xargs/grep) so per-file misses don't
# cascade through `xargs`'s 123 exit and trip `set -e`.
# Pipeline below is wrapped with `|| true` because (a) grep returns non-zero
# if no matches and (b) `set -o pipefail` would otherwise turn that into a
# script-killing exit. We *want* to handle the "no matches" case explicitly
# below.
AVG=$(
    {
        find kernels -mindepth 2 -type f \
            \( -name '*.cu' -o -name '*.hip' -o -name '*.cl' \
               -o -name '*.comp' -o -name '*.metal' \) \
            -exec grep -hE '(extern "C".*void|__kernel void|kernel void) [A-Za-z0-9_]+' {} + 2>/dev/null \
            || true
    } \
    | sed -E 's/.*(extern "C".*void|__kernel void|kernel void) ([A-Za-z0-9_]+).*/\2/' \
    | awk '{s+=length($1); n++} END {if (n) print s/n; else print 0}'
)
if [[ -z "$AVG" || "$AVG" == "0" ]]; then
    KERNEL_FILE_COUNT=$(find kernels -mindepth 2 -type f \
        \( -name '*.cu' -o -name '*.hip' -o -name '*.cl' \
           -o -name '*.comp' -o -name '*.metal' \) 2>/dev/null | wc -l)
    if [[ "$KERNEL_FILE_COUNT" -eq 0 ]]; then
        fail "Symbol-length scan found no kernel sources under kernels/*/* — gate would silently pass; check the layout"
    else
        fail "Symbol-length scan found 0 kernel symbols across $KERNEL_FILE_COUNT source files — gate is silently passing; check the regex"
    fi
elif (( $(echo "$AVG > 40" | bc -l 2>/dev/null || echo 0) )); then
    fail "Average C ABI symbol length is $AVG (limit 40)"
else
    pass "Average C ABI symbol length: $AVG"
fi

# Trademark / brand leakage: scan all C-style sources, the public manifest
# data, and the rendered docs case-insensitively for the OpenFOAM
# Foundation's trademarked terms. Excludes hand-authored prose that is
# explicitly meant to discuss provenance (CONTRIBUTING / CHANGELOG /
# README's Provenance section is allowlisted), the dlpack third-party
# LICENSE, this checker itself, and the Apache LICENSE text.
TRADEMARK_HITS=$(
    {
        find runtime kernels orchestrator python/bindings -type f \
            \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' -o -name '*.h' -o -name '*.hpp' \
               -o -name '*.cu' -o -name '*.hip' -o -name '*.cl' -o -name '*.comp' -o -name '*.metal' \
               -o -name '*.mm' -o -name '*.C' -o -name '*.H' \) 2>/dev/null \
            -exec grep -liE 'interfoam|openfoam' {} + 2>/dev/null
        # Manifest data + rendered docs index — these are emitted by the
        # public-tree generator and should agree with the runtime registry.
        find kernels docs -type f \( -name '*.json' -o -name 'KERNEL_INDEX.md' \
                                    -o -name 'BACKEND_COVERAGE.md' \) 2>/dev/null \
            -exec grep -liE 'interfoam|openfoam' {} + 2>/dev/null
    } \
    | grep -v 'tools/check_generated_content_tells.sh' \
    | sort -u \
    || true
)
if [[ -n "$TRADEMARK_HITS" ]]; then
    fail "Public source/manifest/index files contain OpenFOAM/InterFoam trademark references (case-insensitive):"
    echo "$TRADEMARK_HITS" | sed 's/^/      /' >&2
else
    pass "No OpenFOAM/InterFoam trademark leakage in public source/manifest/index files"
fi

echo
if (( failures > 0 )); then
    echo "FAIL: $failures check(s) failed." >&2
    exit 1
fi
echo "PASS: all acceptance-gate checks succeeded."
