#!/usr/bin/env bash
# Build the Wasm engine and populate the @usex/ternava package(s).
#
# Stage 2 build (wasm-pack as the orchestrator). Tiers × targets:
#   tiers:   mini = d256, base = d384
#   targets: nodejs, bundler → per-package pkg-node/ | pkg-bundler/, routed to
#                              consumers via each package.json "exports" map.
#            web              → per-package pkg-web/ — raw browser glue (./web)
#                              and portable Node/Bun/browser loader (./runtime),
#                              both sharing one wasm file with explicit init().
# All three targets are published npm artifacts; the default matrix builds them all.
# Each build embeds its model and tokenizer through compile-time asset paths.
#
# Usage:
#   bash scripts/build-engine.sh                        # bilingual mini, all targets
#   PKG=packages/base bash scripts/build-engine.sh      # one tier, all targets
#   PKG=packages/base TARGET=bundler bash scripts/build-engine.sh   # one cell
#   WEB_OUT=dist/web TARGET=web bash scripts/build-engine.sh        # web build to a dim-keyed
#                                                        standalone tree instead (CDN use)
#   BIN=path/to/model.bin PKG=... bash scripts/build-engine.sh      # custom bin
#   PROFILE=debug ... / FEATURE=emb_int8 ...            # build variants
#
# See docs-local/tern-bundling.md → Stage 3 for the longer-term plan of
# dropping down to wasm-bindgen-cli directly (single shared wasm per package).

set -euo pipefail

PROFILE="${PROFILE:-release}"
FEATURE="${FEATURE:-emb_int4}"   # current ship target for both tiers
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_DIR="$ROOT/engine"
cd "$ROOT"

# Tier definitions: package dir → packed model .bin to embed.
default_bin_for() {
    case "$1" in
        */mini) echo "$ROOT/training/pack/out/pilot-fa-en-reasoning/model.bin" ;; # bilingual d256
        */base) echo "$ROOT/training/pack/out/d384/model-int4.bin" ;;   # d384
        *)      echo "" ;;
    esac
}

# Model dimension per tier — used to namespace the non-package (web) output tree.
dim_for() {
    case "$1" in
        */mini) echo "d256" ;;
        */base) echo "d384" ;;
        *)      echo "unknown" ;;
    esac
}

PKGS=${PKG:-"packages/mini"}
TARGETS=${TARGET:-"nodejs bundler web"}

build_one() (
    local pkg_dir="$1" target="$2" bin="$3"

    # Output routing per target:
    #   nodejs / bundler → published npm artifacts, per-package (pkg-node / pkg-bundler)
    #   web             → pkg-web, or a standalone tree under WEB_OUT/<dim>/.
    local target_dir
    case "$target" in
        nodejs)  target_dir="$ROOT/$pkg_dir/pkg-node" ;;
        bundler) target_dir="$ROOT/$pkg_dir/pkg-bundler" ;;
        web)     if [[ -n "${WEB_OUT:-}" ]]; then
                     target_dir="$ROOT/$WEB_OUT/$(dim_for "$pkg_dir")"
                 else
                     target_dir="$ROOT/$pkg_dir/pkg-web"
                 fi ;;
        *) echo "ERROR: unknown target '$target' (want: nodejs | bundler | web)" >&2; return 1 ;;
    esac

    echo ""
    echo "── Building $pkg_dir ($target, $PROFILE, --features $FEATURE) ──"
    echo "   model: $bin"
    local tokenizer="${TOKENIZER:-${bin%.bin}.tokenizer.json}"
    if [[ -z "${TOKENIZER:-}" && ! -f "$tokenizer" ]]; then
        tokenizer="$ENGINE_DIR/assets/tokenizer.json"
    fi
    python3 "$ROOT/scripts/check-model.py" "$bin" "$tokenizer"
    export TERN_MODEL_PATH="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"
    export TERN_TOKENIZER_PATH="$(cd "$(dirname "$tokenizer")" && pwd)/$(basename "$tokenizer")"

    cd "$ENGINE_DIR"
    if [[ "$PROFILE" == "release" ]]; then
        wasm-pack build --target "$target" --release --features "$FEATURE"
        if command -v wasm-opt >/dev/null 2>&1; then
            wasm-opt -Oz pkg/tern_engine_bg.wasm -o pkg/tern_engine_bg.wasm
        else
            echo "WARNING: wasm-opt not found — skipping size optimization"
        fi
    else
        wasm-pack build --target "$target" --features "$FEATURE"
    fi

    # Copy the glue + wasm + types — skip wasm-pack's auto package.json,
    # README, and .gitignore (the package ships its own metadata).
    rm -rf "$target_dir"
    mkdir -p "$target_dir"
    find pkg -maxdepth 1 -name 'tern_engine*' -exec cp {} "$target_dir/" \;

    # The web-target glue is ESM, but the package root declares "type":
    # "commonjs" — without a nearer package.json Node would misparse the glue
    # as CJS (issue #6 follow-up). Scope the module type to this directory.
    if [[ "$target" != "nodejs" ]]; then
        printf '{ "type": "module" }\n' > "$target_dir/package.json"
    fi
    if [[ "$target" == "web" ]]; then
        cp "$ROOT/engine/runtime/index.mjs" "$ROOT/engine/runtime/index.d.mts" "$target_dir/"
    fi

    local wasm_bytes wasm_mb
    wasm_bytes=$(wc -c <"$target_dir/tern_engine_bg.wasm")
    wasm_mb=$(awk "BEGIN {printf \"%.2f\", $wasm_bytes / 1024 / 1024}")
    echo "   → ${target_dir#"$ROOT"/}/tern_engine_bg.wasm = ${wasm_mb} MB"
)

for pkg_dir in $PKGS; do
    bin="${BIN:-$(default_bin_for "$pkg_dir")}"
    if [[ -z "$bin" || ! -f "$bin" ]]; then
        echo "ERROR: no model .bin for $pkg_dir (looked for: ${bin:-<none>})" >&2
        echo "       run training/pack/pack.py first, or pass BIN=..." >&2
        exit 1
    fi
    for target in $TARGETS; do
        build_one "$pkg_dir" "$target" "$bin"
    done
done

echo ""
echo "Done."
