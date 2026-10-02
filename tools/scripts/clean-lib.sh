# shellcheck shell=bash
# clean-lib.sh
#
# The shared half of every folder's clean.sh. A folder that produces build
# output owns a clean.sh that sources this file, names what the folder
# produces, and hands over to clean_run:
#
#   #!/usr/bin/env bash
#   # Flutter example app.
#   . "$(dirname "$0")/../../../tools/scripts/clean-lib.sh"
#   clean_command flutter clean
#   clean_paths build .dart_tool ios/Pods
#   clean_run "$@"
#
# Paths are relative to the folder and may be globs. Without --apply, clean_run
# lists each one that exists with its size. With --apply it runs the command
# (skipped when it is not installed, or when nothing was built), then deletes
# whatever paths remain. A path is only touched when git ignores it and tracks
# nothing inside it: a wrong entry is reported, never deleted.
#
# The root clean.sh runs every clean.sh git tracks, in parallel.
# It sets CLEAN_KB_FILE, where clean_run writes its total instead of a summary.
set -euo pipefail

CLEAN_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEAN_ROOT="$(git -C "$CLEAN_DIR" rev-parse --show-toplevel)"
CLEAN_APPLY=0
CLEAN_TOTAL_KB=0
CLEAN_PATHS=()
CLEAN_COMMAND=()

# clean_paths <path>...: output paths of this folder, relative to it.
clean_paths() { CLEAN_PATHS+=("$@"); }

# clean_command <cmd> [args]...: a tool that cleans more than the paths list
# (flutter clean also drops generated platform files). Runs in the folder.
clean_command() { CLEAN_COMMAND=("$@"); }

# clean_run [--apply]: report or delete this folder's outputs.
clean_run() {
  clean_args "$@"
  clean_folder
  clean_summary
}

clean_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --apply) CLEAN_APPLY=1 ;;
      -h | --help) clean_usage ;;
      *)
        echo "$(display "$0"): unknown argument: $1 (see --help)" >&2
        exit 2
        ;;
    esac
    shift
  done
}

# clean_header: print the calling script's header comment.
clean_header() {
  awk 'NR > 1 && /^# shellcheck / { next }
       NR > 1 && /^#/ { sub(/^# ?/, ""); print; next }
       NR > 1 { exit }' "$0"
}

clean_usage() {
  clean_header
  echo
  echo "Usage: $(display "$0") [--apply]   (dry run without --apply)"
  exit 0
}

# clean_folder: report, and with --apply delete, the paths of CLEAN_DIR. A
# glob matching several paths reports as one line.
clean_folder() {
  local pattern path matched kb i
  local found=() skipped=() reasons=() sizes=() labels=()
  [ "${#CLEAN_PATHS[@]}" -gt 0 ] || return 0
  for pattern in "${CLEAN_PATHS[@]}"; do
    matched=0
    kb=0
    # Unquoted on purpose: expand globs. A pattern matching nothing stays
    # literal and fails the -e test.
    for path in "$CLEAN_DIR"/$pattern; do
      [ -e "$path" ] || continue
      if is_checkout "$path"; then
        skipped+=("$path")
        reasons+=("a git checkout, not build output")
        continue
      fi
      if ! is_disposable "$path"; then
        skipped+=("$path")
        reasons+=("git does not ignore it, so clean.sh should not list it")
        continue
      fi
      found+=("$path")
      matched=$((matched + 1))
      kb=$((kb + $(kb_of "$path")))
    done
    [ "$matched" -gt 0 ] || continue
    sizes+=("$kb")
    if [ "$matched" = 1 ]; then
      labels+=("$(display "$path")")
    else
      labels+=("$(display "$CLEAN_DIR/$pattern") ($matched matches)")
    fi
  done
  [ "${#found[@]}" -gt 0 ] || [ "${#skipped[@]}" -gt 0 ] || return 0

  heading "$(display "$CLEAN_DIR")"
  i=0
  while [ "$i" -lt "${#sizes[@]}" ]; do
    report "${sizes[$i]}" "${labels[$i]}"
    i=$((i + 1))
  done
  i=0
  while [ "$i" -lt "${#skipped[@]}" ]; do
    skip "${skipped[$i]}" "${reasons[$i]}"
    i=$((i + 1))
  done
  [ "${#found[@]}" -gt 0 ] || return 0
  if [ "${#CLEAN_COMMAND[@]}" -gt 0 ]; then
    if [ "$CLEAN_APPLY" = 0 ]; then
      note "then: ${CLEAN_COMMAND[*]}"
    elif ! command -v "${CLEAN_COMMAND[0]}" >/dev/null 2>&1; then
      note "${CLEAN_COMMAND[0]} is not installed; deleting the paths directly"
    elif ! (cd "$CLEAN_DIR" && "${CLEAN_COMMAND[@]}") </dev/null >/dev/null 2>&1
    then
      note "${CLEAN_COMMAND[*]} failed; deleting the paths directly"
    fi
  fi
  if [ "$CLEAN_APPLY" = 1 ]; then
    for path in "${found[@]}"; do delete_path "$path"; done
  fi
}

clean_summary() {
  if [ -n "${CLEAN_KB_FILE:-}" ]; then
    echo "$CLEAN_TOTAL_KB" >"$CLEAN_KB_FILE"
  elif [ "$CLEAN_APPLY" = 1 ]; then
    printf '\nFreed %s.\n' "$(human "$CLEAN_TOTAL_KB")"
  else
    printf '\nWould free %s. Re-run with --apply to delete.\n' \
      "$(human "$CLEAN_TOTAL_KB")"
  fi
}

# --- helpers ---------------------------------------------------------------

kb_of() {
  local kb
  kb="$(du -sk "$1" 2>/dev/null | awk '{ print $1 }')" || true
  echo "${kb:-0}"
}

human() {
  awk -v k="$1" 'BEGIN {
    split("K M G T", unit, " "); i = 1
    while (k >= 1024 && i < 4) { k /= 1024; i++ }
    printf(i == 1 ? "%d%s" : "%.1f%s", k, unit[i])
  }'
}

display() {
  case "$1" in
    "$CLEAN_ROOT") echo "." ;;
    *) echo "${1#"$CLEAN_ROOT"/}" ;;
  esac
}

heading() { printf '\n%s\n' "$1"; }

note() { printf '  %8s  %s\n' "" "$1"; }

skip() { printf '  %8s  %s (%s)\n' "skip" "$(display "$1")" "$2"; }

# report <kb> <label>: print one candidate and add it to the total.
report() {
  CLEAN_TOTAL_KB=$((CLEAN_TOTAL_KB + $1))
  printf '  %8s  %s\n' "$(human "$1")" "$2"
}

# account <path> [kb]: report one path, measuring it unless told its size.
account() { report "${2:-$(kb_of "$1")}" "$(display "$1")"; }

# delete_path <path>: rm -rf, retrying writable for read-only trees (Bazel, Go).
delete_path() {
  rm -rf "$1" 2>/dev/null && return 0
  chmod -R u+w "$1" 2>/dev/null || true
  rm -rf "$1"
}

# is_checkout <path>: <path> is the top of a git checkout or worktree, which
# holds work of its own whatever its name. Only the top: an output tree can
# hold a .git deeper down (cargo's target/ keeps a llama.cpp clone).
is_checkout() { [ -e "$1/.git" ]; }

# is_disposable <path>: git ignores <path> and tracks nothing inside it, so
# deleting it cannot lose committed or pending work. check-ignore reads the
# index: a directory holding a tracked file does not count as ignored.
is_disposable() {
  git -C "$(dirname "$1")" check-ignore -q -- "$(basename "$1")" 2>/dev/null
}
