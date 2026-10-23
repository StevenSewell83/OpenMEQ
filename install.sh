#!/usr/bin/env bash
#
# install.sh -- single-command installer for OpenMEQ (docs/PLAN.md §2,
# docs/ARCHITECTURE.md §7).
#
# Builds, in order:
#   1. A Python 3.10 venv (.venv) via uv.
#   2. OpenMEQ + its runtime deps (ObsPy, TensorFlow 2.11, PyQt5, ObsPyck, ...)
#      with `uv pip install -e .` (or `-e .[dev]` with --dev).
#   3. PhaseNet (git clone at a pinned commit) into external/PhaseNet, with
#      its pretrained model weights.
#   4. NonLinLoc, built from source with CMake, binaries copied to
#      external/bin/.
#   5. hypoDD 2.1 (3D), unpacked from HYPODD.zip and built with make,
#      binaries copied to external/bin/.
#   6. A smoke check: imports succeed, binaries run without a missing-library
#      error, and hypoDD's bundled example 1 relocates end-to-end.
#
# Usage:
#   ./install.sh                    # full install (steps 1-6 above)
#   ./install.sh --dev              # also installs the [dev] extra (pytest, ruff, ...)
#   ./install.sh --skip-external    # steps 1-2 only: venv + `pip install -e .`,
#                                    # no network clones of PhaseNet/NonLinLoc,
#                                    # no hypoDD build, no smoke check. For CI
#                                    # or offline dev use.
#   ./install.sh --help             # this message
#
# System prerequisites (checked below; the script fails fast naming whatever
# is missing rather than dying deep inside a build step):
#   - git, unzip        (fetch PhaseNet/NonLinLoc sources, unpack HYPODD.zip)
#   - cmake, gcc        (build NonLinLoc)
#   - gfortran, make    (build hypoDD/ph2dt)
#   - uv                (https://docs.astral.sh/uv/; ~/.local/bin/uv or on PATH.
#                        install.sh does NOT install uv itself -- if it's
#                        missing you'll get an install hint and a non-zero exit.)
#
# Re-running this script is safe (idempotent): an existing .venv is reused/
# recreated by `uv venv`, an existing external/PhaseNet or external/NonLinLoc
# clone is fetched and checked out to the pinned commit rather than re-cloned,
# and an existing external/HYPODD is rebuilt in place rather than re-unzipped.

set -euo pipefail

# --- Pinned versions -------------------------------------------------------
#
# PhaseNet (AI4EPS/PhaseNet) is not on PyPI, so we clone it at a pinned commit
# rather than track a branch HEAD, for a reproducible install. Checked
# 2026-09-28: this commit's requirements.txt pins tensorflow==2.11.0 and
# numpy==1.26.4 (matching this project's own pins -- no conflict to resolve),
# and model/190703-214543 is present as real checked-in weight files (not
# LFS pointers).
PHASENET_REPO_URL="https://github.com/AI4EPS/PhaseNet"
PHASENET_COMMIT="62005c6195638f88d077f8870ea895971062715a"
PHASENET_MODEL_SUBDIR="model/190703-214543"

# NonLinLoc (alomax/NonLinLoc) is not on PyPI/conda-forge either. Pinned for
# the same reproducibility reason. Verified 2026-09-28: `cmake -S src -B
# build && cmake --build build -j$(nproc)` completes in a few seconds with
# gcc 12 / gfortran 11, producing (among others) src/bin/{NLLoc,Vel2Grid,
# Vel2Grid3D,Grid2Time}.
NLLOC_REPO_URL="https://github.com/alomax/NonLinLoc"
NLLOC_COMMIT="7a62fa46eff682083352a2ca28e63a4f415d941b"

# Binaries later tasks (M4.1, M5.1, M5.2/M5.3, M9.x) depend on.
NLLOC_REQUIRED_BINS=(NLLoc Vel2Grid Vel2Grid3D Grid2Time)
HYPODD_REQUIRED_BINS=(hypoDD ph2dt)

# --- Paths -------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
EXTERNAL_DIR="$REPO_ROOT/external"
BIN_DIR="$EXTERNAL_DIR/bin"
LOG_FILE="$REPO_ROOT/install.log"
HYPODD_ZIP="$REPO_ROOT/HYPODD.zip"

# --- Logging (stdout/stderr tee'd to install.log, truncated each run) ------
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    printf '\n[install.sh] %s\n' "$*"
}

die() {
    printf '\n[install.sh] ERROR: %s\n' "$*" >&2
    exit 1
}

# --- Prerequisite checks ----------------------------------------------------

resolve_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    if [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
        return
    fi
    die "uv not found on PATH or at ~/.local/bin/uv. install.sh does not" \
        "install uv itself -- install it first, e.g.:" \
        $'\n\n    curl -LsSf https://astral.sh/uv/install.sh | sh\n\n' \
        "See https://docs.astral.sh/uv/getting-started/installation/"
}

require_tools() {
    local missing=()
    local tool
    for tool in "$@"; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        die "missing required tool(s): ${missing[*]}. Install them and re-run install.sh."
    fi
}

# --- Compilers ---------------------------------------------------------------
#
# NonLinLoc (C) and hypoDD (C + Fortran, linked by gfortran) must be built by
# one consistent toolchain. A bare `gcc` from PATH is not good enough: an
# active conda env puts its own gcc ahead of the system one and exports
# CC/CFLAGS/LDFLAGS (x86_64-conda-linux-gnu-cc, -isystem <conda>/include,
# -rpath <conda>/lib), which CMake picks up -- NLLoc then ends up with the
# conda lib dir baked into its RPATH, and hypoDD mixes conda-gcc C objects
# with system-gfortran Fortran objects.
#
# So: take gfortran from PATH (or $OPENMEQ_FC), take the gcc that sits next
# to it in the same install (or $OPENMEQ_CC), and run the builds with the
# inherited compiler variables removed (see build_env).
select_compilers() {
    FC="${OPENMEQ_FC:-$(command -v gfortran)}"
    [ -x "$FC" ] || die "Fortran compiler not executable: $FC"

    if [ -n "${OPENMEQ_CC:-}" ]; then
        CC="$OPENMEQ_CC"
    else
        local fc_dir fc_ver cand
        fc_dir="$(dirname "$(readlink -f "$FC")")"
        fc_ver="$("$FC" -dumpversion)"
        CC=""
        for cand in "$fc_dir/gcc-${fc_ver%%.*}" "$fc_dir/gcc"; do
            if [ -x "$cand" ]; then
                CC="$cand"
                break
            fi
        done
        [ -n "$CC" ] || die "no gcc found alongside $FC (looked in $fc_dir). Set OPENMEQ_CC to the matching C compiler."
    fi
    [ -x "$CC" ] || die "C compiler not executable: $CC"

    local cc_ver fc_ver_full
    cc_ver="$("$CC" -dumpfullversion -dumpversion)"
    fc_ver_full="$("$FC" -dumpfullversion -dumpversion)"
    log "compilers: CC=$CC ($cc_ver), FC=$FC ($fc_ver_full)"
    if [ "${cc_ver%%.*}" != "${fc_ver_full%%.*}" ]; then
        log "WARNING: C and Fortran compiler major versions differ ($cc_ver vs $fc_ver_full)"
    fi
}

# Runs "$@" with compiler/flag variables inherited from the caller's shell
# (e.g. a conda activation) removed, so only CC/FC from select_compilers apply.
build_env() {
    env -u CC -u CXX -u FC -u F77 -u F90 -u CPP \
        -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u FFLAGS -u FCFLAGS -u LDFLAGS \
        -u CONDA_BUILD_SYSROOT \
        "$@"
}

# --- Step 1: venv ------------------------------------------------------------

install_venv() {
    log "venv: ensuring Python 3.10 is available to uv"
    if ! "$UV" python list --only-installed 2>/dev/null | grep -q '^cpython-3\.10'; then
        "$UV" python install 3.10
    fi
    log "venv: creating $VENV_DIR (uv venv --python 3.10)"
    "$UV" venv --python 3.10 "$VENV_DIR"
}

# --- Step 2: OpenMEQ + deps --------------------------------------------------

install_openmeq() {
    local extra="."
    if [ "$DEV_INSTALL" -eq 1 ]; then
        extra=".[dev]"
    fi
    log "deps: uv pip install -e $extra (ObsPy, TensorFlow 2.11, PyQt5, ObsPyck, PyYAML, ...)"
    "$UV" pip install --python "$VENV_PYTHON" -e "$extra"
}

# --- Step 3: PhaseNet ---------------------------------------------------------

install_phasenet() {
    local dir="$EXTERNAL_DIR/PhaseNet"
    log "PhaseNet: pinned commit $PHASENET_COMMIT"
    if [ -d "$dir/.git" ]; then
        log "PhaseNet: external/PhaseNet already present; fetching and checking out pinned commit"
        if ! git -C "$dir" checkout --quiet "$PHASENET_COMMIT" 2>/dev/null; then
            git -C "$dir" fetch --quiet origin "$PHASENET_COMMIT"
            git -C "$dir" checkout --quiet "$PHASENET_COMMIT"
        fi
    else
        log "PhaseNet: cloning $PHASENET_REPO_URL"
        git clone --quiet "$PHASENET_REPO_URL" "$dir"
        git -C "$dir" checkout --quiet "$PHASENET_COMMIT"
    fi

    local model_dir="$dir/$PHASENET_MODEL_SUBDIR"
    [ -d "$model_dir" ] || die "PhaseNet model weights missing at $model_dir after checkout of $PHASENET_COMMIT"

    # PhaseNet's setup.py reads its own requirements.txt as install_requires,
    # so this alone pulls in tensorflow==2.11.0, numpy==1.26.4 (already
    # satisfied by pyproject.toml's own pins -- no reinstall) plus its
    # PhaseNet-only extras (scikit-learn, fastapi, uvicorn, ...).
    log "PhaseNet: uv pip install -e external/PhaseNet"
    "$UV" pip install --python "$VENV_PYTHON" -e "$dir"
}

# --- Step 4: NonLinLoc ---------------------------------------------------------

build_nonlinloc() {
    local dir="$EXTERNAL_DIR/NonLinLoc"
    log "NonLinLoc: pinned commit $NLLOC_COMMIT"
    if [ -d "$dir/.git" ]; then
        log "NonLinLoc: external/NonLinLoc already present; fetching and checking out pinned commit"
        if ! git -C "$dir" checkout --quiet "$NLLOC_COMMIT" 2>/dev/null; then
            git -C "$dir" fetch --quiet origin "$NLLOC_COMMIT"
            git -C "$dir" checkout --quiet "$NLLOC_COMMIT"
        fi
    else
        log "NonLinLoc: cloning $NLLOC_REPO_URL"
        git clone --quiet "$NLLOC_REPO_URL" "$dir"
        git -C "$dir" checkout --quiet "$NLLOC_COMMIT"
    fi

    # NonLinLoc's CMakeLists.txt lives under src/, not the repo root. The
    # build dir is configured from scratch each run: CMake caches the
    # compiler and flags it first saw, so a stale cache would keep using
    # them regardless of CC below.
    #
    # The C standard is pinned to gnu17: GCC 15 defaults to C23, where an
    # empty parameter list `int f();` means `int f(void)`, and NonLinLoc's
    # K&R-style prototypes (e.g. vector/vector.h) then conflict with their
    # definitions ("conflicting types for 'cross_product_3d'").
    log "NonLinLoc: cmake configure + build"
    rm -rf "$dir/build"
    build_env cmake -S "$dir/src" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_C_STANDARD=17 -DCMAKE_C_EXTENSIONS=ON
    build_env cmake --build "$dir/build" -j"$(nproc)"

    local built_bin="$dir/src/bin"
    [ -d "$built_bin" ] || die "NonLinLoc build did not produce $built_bin -- cmake target layout may have changed upstream"

    mkdir -p "$BIN_DIR"
    log "NonLinLoc: copying binaries from ${built_bin#"$REPO_ROOT"/} to external/bin/"
    find "$built_bin" -maxdepth 1 -type f -perm -u+x -exec cp -f {} "$BIN_DIR/" \;

    local b
    for b in "${NLLOC_REQUIRED_BINS[@]}"; do
        [ -x "$BIN_DIR/$b" ] || die "expected NonLinLoc binary external/bin/$b missing after build"
    done
}

# --- Step 5: hypoDD -------------------------------------------------------------

build_hypodd() {
    local dir="$EXTERNAL_DIR/HYPODD"
    [ -f "$HYPODD_ZIP" ] || die "HYPODD.zip not found at repo root ($HYPODD_ZIP)"

    if [ -d "$dir" ]; then
        log "hypoDD: external/HYPODD already extracted; skipping unzip"
    else
        log "hypoDD: unzipping HYPODD.zip into external/"
        unzip -q "$HYPODD_ZIP" -d "$EXTERNAL_DIR"
    fi
    [ -d "$dir/src" ] || die "unexpected HYPODD.zip layout: $dir/src not found"

    # The Makefiles hardcode CC = gcc / FC = gfortran (resolved via PATH);
    # command-line assignments override those and reach the sub-makes
    # through MAKEFLAGS.
    log "hypoDD: make clean && make (src/hypoDD, src/ph2dt)"
    (cd "$dir/src" && build_env make clean && build_env make CC="$CC" FC="$FC")

    mkdir -p "$BIN_DIR"
    [ -x "$dir/src/hypoDD/hypoDD" ] || die "hypoDD build did not produce src/hypoDD/hypoDD"
    [ -x "$dir/src/ph2dt/ph2dt" ] || die "hypoDD build did not produce src/ph2dt/ph2dt"
    cp -f "$dir/src/hypoDD/hypoDD" "$BIN_DIR/hypoDD"
    cp -f "$dir/src/ph2dt/ph2dt" "$BIN_DIR/ph2dt"
}

# --- Step 6: smoke check ---------------------------------------------------------

# Runs a binary that unconditionally prompts on stdin (old Fortran/C codes
# with no real --help), with stdin closed, and checks the combined output for
# an expected banner substring. These programs exit non-zero on the resulting
# EOF/usage condition -- that's expected and NOT the failure mode we check
# for; a *missing-library* failure looks different (a dynamic loader message
# such as "error while loading shared libraries") and is what we reject.
run_and_check_output() {
    local expect="$1"
    shift
    local out
    out="$("$@" < /dev/null 2>&1 || true)"
    if printf '%s' "$out" | grep -qi 'error while loading shared librar\|cannot open shared object file'; then
        die "$1 failed with a missing-library error:"$'\n'"$out"
    fi
    if ! printf '%s' "$out" | grep -qi "$expect"; then
        die "$1 did not print the expected output ('$expect'):"$'\n'"$out"
    fi
}

smoke_check() {
    log "smoke check: imports"
    "$VENV_PYTHON" -c "import openmeq, obspy, numpy, tensorflow, PyQt5, obspyck" \
        || die "smoke check: python import failed"

    log "smoke check: NonLinLoc binaries run without a missing-library error"
    run_and_check_output "Usage:" "$BIN_DIR/NLLoc"
    run_and_check_output "Usage:" "$BIN_DIR/Vel2Grid"
    run_and_check_output "Usage:" "$BIN_DIR/Grid2Time"

    log "smoke check: hypoDD binaries run without a missing-library error"
    # hypoDD/ph2dt write a <name>.log file into the current directory on
    # startup (before even reading stdin), so run them from external/
    # (already gitignored) rather than the repo root.
    ( cd "$EXTERNAL_DIR" && run_and_check_output "starting hypoDD" "$BIN_DIR/hypoDD" )
    ( cd "$EXTERNAL_DIR" && run_and_check_output "starting ph2dt" "$BIN_DIR/ph2dt" )

    log "smoke check: hypoDD bundled example 1 (ph2dt -> hypoDD) end-to-end"
    local example_dir="$EXTERNAL_DIR/HYPODD/examples/example1"
    [ -d "$example_dir" ] || die "hypoDD example1 not found at $example_dir"

    ( cd "$example_dir" \
      && rm -f dt.ct event.dat event.sel station.sel ph2dt.log hypoDD.reloc hypoDD.loc hypoDD.res hypoDD.src hypoDD.sta hypoDD.log \
      && "$BIN_DIR/ph2dt" ph2dt.inp )
    [ -f "$example_dir/dt.ct" ] || die "hypoDD example1: ph2dt did not produce dt.ct"

    ( cd "$example_dir" && "$BIN_DIR/hypoDD" hypoDD.inp )
    [ -s "$example_dir/hypoDD.reloc" ] || die "hypoDD example1: hypoDD did not produce a non-empty hypoDD.reloc"
    if grep -qi 'fatal' "$example_dir/hypoDD.log" 2>/dev/null; then
        die "hypoDD example1: hypoDD.log reports a fatal error"
    fi
    log "smoke check: hypoDD example1 produced $(wc -l < "$example_dir/hypoDD.reloc" | tr -d ' ') relocated events in hypoDD.reloc"
}

# --- Usage / argument parsing -----------------------------------------------

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

DEV_INSTALL=0
SKIP_EXTERNAL=0
for arg in "$@"; do
    case "$arg" in
        --dev) DEV_INSTALL=1 ;;
        --skip-external) SKIP_EXTERNAL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $arg (see ./install.sh --help)" ;;
    esac
done

# --- Main ------------------------------------------------------------------

main() {
    log "OpenMEQ installer starting (repo root: $REPO_ROOT)"

    UV="$(resolve_uv)"
    log "using uv: $UV ($("$UV" --version))"

    require_tools git

    install_venv
    install_openmeq

    if [ "$SKIP_EXTERNAL" -eq 1 ]; then
        log "--skip-external: skipping PhaseNet/NonLinLoc/hypoDD and the smoke check"
        print_summary
        return
    fi

    require_tools git unzip cmake gcc gfortran make
    select_compilers

    mkdir -p "$EXTERNAL_DIR" "$BIN_DIR"
    install_phasenet
    build_nonlinloc
    build_hypodd
    smoke_check

    print_summary
}

print_summary() {
    log "install.sh finished successfully"
    echo "  venv:      $VENV_DIR (Python 3.10, $([ "$DEV_INSTALL" -eq 1 ] && echo "dev" || echo "runtime") deps)"
    if [ "$SKIP_EXTERNAL" -eq 1 ]; then
        echo "  external:  skipped (--skip-external)"
    else
        echo "  PhaseNet:  $EXTERNAL_DIR/PhaseNet @ $PHASENET_COMMIT (model: $PHASENET_MODEL_SUBDIR)"
        echo "  NonLinLoc: $EXTERNAL_DIR/NonLinLoc @ $NLLOC_COMMIT -> $BIN_DIR/{${NLLOC_REQUIRED_BINS[*]}}"
        echo "  hypoDD:    $EXTERNAL_DIR/HYPODD -> $BIN_DIR/{${HYPODD_REQUIRED_BINS[*]}}"
        echo "  smoke check: passed (imports, binaries, hypoDD example1 relocation)"
    fi
    echo "  log:       $LOG_FILE"
    echo
    echo "Activate with: source .venv/bin/activate"
}

main
