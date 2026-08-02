#!/usr/bin/env bash

# Research is the default mode.  Production files are only touched by the
# explicit --publish command and a passing, user-supplied approval record.
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/rl-portfolio-allocator"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

log() { printf '[rlpa] %s\n' "$*"; }
die() { printf '[rlpa] ERROR: %s\n' "$*" >&2; exit 1; }

require_credentials() {
    [[ -n "${PANDA_DATA_USERNAME:-}" && -n "${PANDA_DATA_PASSWORD:-}" ]] || \
        die 'PANDA_DATA_USERNAME and PANDA_DATA_PASSWORD are required for data generation'
}

run_features() {
    require_credentials
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.features)
}

run_market_state() {
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.market_state)
}

run_coverage() {
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.check_data_coverage \
        --features data/features.parquet \
        --index data/index_returns.parquet \
        --json artifacts/state/data_coverage.json)
}

run_tests() { (cd "$WORK_DIR" && "$PYTHON_BIN" -m pytest -q); }

run_walk_forward() {
    local mode="$1"
    local marker
    marker="$(mktemp)"
    if [[ "$mode" == "--full" && -n "${RLPA_RUN_ID:-}" ]]; then
        (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.walk_forward "$mode" \
            --output-root artifacts/walk_forward --run-id "$RLPA_RUN_ID")
    else
        (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.walk_forward "$mode" \
            --output-root artifacts/walk_forward)
    fi
    LAST_RUN_ROOT="$(find "$WORK_DIR/artifacts/walk_forward" -mindepth 2 -maxdepth 2 \
        -type f -name gates.json -newer "$marker" -print -exec dirname {} \; | tail -n 1)"
    rm -f "$marker"
    [[ -n "$LAST_RUN_ROOT" ]] || die "walk-forward did not produce a gates.json"
    log "walk-forward run: $LAST_RUN_ROOT"
}

run_research_gates() {
    local run_root="$1"
    local allow_failed="${2:-false}"
    local gates="${WORK_DIR}/${run_root}/gates.json"
    [[ -f "$gates" ]] || die "research gates report missing: $gates"
    "$PYTHON_BIN" - "$gates" "$allow_failed" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("research_ok") is not True:
    failed = [g.get("name", "unknown") for g in report.get("gates", []) if not g.get("passed")]
    message = "research gate failed: " + ", ".join(failed or ["missing research_ok=true"])
    if sys.argv[2] == "true":
        print("WARNING: " + message + "; publication remains disabled")
    else:
        raise SystemExit(message)
print(f"research gates passed: {path}")
PY
}

run_stress() {
    local run_root="${1:-}"
    local method="${run_root}/summary.json"
    local report="${run_root}/stress.json"
    [[ -f "$method" ]] || die "walk-forward summary missing: $method"
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.stress_test \
        --method "$method" --report "$report")
}

run_publish() {
    local approval="$1"
    [[ -n "$approval" ]] || die '--publish requires --approval APPROVAL_JSON'
    if [[ "$approval" != /* ]]; then
        approval="$PWD/$approval"
    fi
    local approval_dir
    approval_dir="$(dirname "$approval")"
    [[ -d "$approval_dir" ]] || die "approval directory missing: $approval_dir"
    approval="$(cd "$approval_dir" && pwd)/$(basename "$approval")"
    [[ -f "$approval" ]] || die "approval file missing: $approval"
    "$PYTHON_BIN" - "$approval" <<'PY'
import json
import pathlib
import sys

approval = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if approval.get("research_ok") is not True:
    raise SystemExit("approval research_ok is not true")
if approval.get("run_mode") != "full":
    raise SystemExit("publish requires approval from a full walk-forward run")
if approval.get("fold_count", 0) < 3 or approval.get("seed_count", 0) < 5:
    raise SystemExit("approval does not contain the complete fold/seed run")
PY
    mkdir -p "$WORK_DIR/checkpoints/candidate-production"
    local candidate_dir
    candidate_dir="$(mktemp -d "$WORK_DIR/checkpoints/candidate-production/rlpa.XXXXXX")"
    local production_dir="$WORK_DIR/../rl-portfolio-allocator-production/data"
    mkdir -p "$candidate_dir"
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.allocate --retrain \
        --approval "$approval" --candidate-dir "$candidate_dir")
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.validate \
        --path "$candidate_dir/allocations.parquet")
    (cd "$WORK_DIR" && "$PYTHON_BIN" -c \
        'from scripts.allocate import atomic_publish; import sys; atomic_publish(sys.argv[1], sys.argv[2])' \
        "$candidate_dir" "$production_dir")
    (cd "$WORK_DIR" && "$PYTHON_BIN" -m scripts.validate \
        --path "$production_dir/allocations.parquet")
    log "production bundle published: $production_dir"
}

run_research() {
    local mode="$1"
    local wf_mode="$2"
    run_features
    run_market_state
    run_coverage
    run_tests
    run_walk_forward "$wf_mode"
    local gate_status=0
    if [[ "$mode" == "smoke" ]]; then
        run_research_gates "${LAST_RUN_ROOT#"$WORK_DIR/"}" true || gate_status=$?
    else
        run_research_gates "${LAST_RUN_ROOT#"$WORK_DIR/"}" false || gate_status=$?
    fi
    run_stress "$LAST_RUN_ROOT"
    return "$gate_status"
}

usage() {
    cat <<'EOF'
Usage: ./run_pipeline.sh [--all | --research-smoke | --research-full | --publish --approval PATH]

Research (fail-closed):
  --all             Alias for --research-smoke; never publishes production files.
  --research-smoke  features -> market_state -> coverage -> pytest -> walk-forward smoke -> gate -> stress.
  --research-full   Same sequence with all configured folds and seeds.

Production (explicit approval required):
  --publish --approval PATH
                    retrain production, then validate allocations. The approval
                    must be emitted by a passing full walk-forward run.

  --help            Show this help without credentials or data access.
EOF
}

main() {
    local command="${1:---help}"
    case "$command" in
        --help|-h) usage ;;
        --all|--research-smoke)
            [[ $# -eq 1 ]] || die "$command does not accept extra arguments"
            run_research smoke --smoke
            ;;
        --research-full)
            [[ $# -eq 1 ]] || die '--research-full does not accept extra arguments'
            run_research full --full
            ;;
        --publish)
            [[ $# -eq 3 && "$2" == "--approval" ]] || die 'usage: --publish --approval APPROVAL_JSON'
            run_publish "$3"
            ;;
        *) usage >&2; die "unknown option: $command" ;;
    esac
}

main "$@"
