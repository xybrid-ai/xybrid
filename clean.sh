#!/usr/bin/env bash
# clean.sh
#
# Reclaim disk in this checkout (typically a Conductor workspace) by deleting
# what a build or install recreates. Dry run by default: it lists every
# candidate with its size. Pass --apply to delete. `just clean` runs it.
#
# Every folder that produces build output owns a clean.sh naming its outputs
# (see tools/scripts/clean-lib.sh). This script runs every clean.sh git tracks
# in parallel, and prints their reports in path order. It also owns:
#   .               the cargo target/ and SwiftPM .build/.
#   .context/       agent scratch: any target/, node_modules/, Pods/, build/...
#                   inside it (it holds throwaway apps and release checkouts,
#                   which have no clean.sh of their own). Inside a checkout
#                   with its own .git, only what that checkout ignores.
#   --worktrees     nested git worktrees under .context/ that are clean and
#                   whose HEAD is on a remote branch or in a merged PR (asked
#                   of `gh`, when installed). Removed by `git worktree remove`.
#   --bazel         this workspace's Bazel output base, via `bazel clean
#                   --expunge`. It lives outside the workspace, so deleting
#                   the workspace folder never frees it.
#   --bazel-orphans Bazel output bases whose workspace no longer exists: the
#                   ones archived workspaces and deleted temp checkouts leave.
#   --all           every tier above.
#
# Stop builds running in the workspace before --apply.
#
# Usage: clean.sh [--apply] [--worktrees] [--bazel] [--bazel-orphans] [--all]
# Env:   BAZEL            Bazel binary for --bazel (default: bazelisk or bazel)
#        BAZEL_USER_ROOT  directory holding output bases (default: derived)
# shellcheck source=tools/scripts/clean-lib.sh
. "$(dirname "$0")/tools/scripts/clean-lib.sh"

clean_paths target .build

# Names that only ever hold build or install output, swept inside .context/.
# `build` and `dist` are generic, which is why is_scratch_output vets each hit.
SCRATCH_OUTPUT_NAMES="target node_modules Pods build .gradle .cxx .next
  .dart_tool .build DerivedData dist .expo bazel-disk-cache"

WORKTREES=0
BAZEL_BASE=0
BAZEL_ORPHANS=0
REMOVED_WORKTREES=""

args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --worktrees) WORKTREES=1 ;;
    --bazel) BAZEL_BASE=1 ;;
    --bazel-orphans) BAZEL_ORPHANS=1 ;;
    --all) WORKTREES=1; BAZEL_BASE=1; BAZEL_ORPHANS=1 ;;
    -h | --help) clean_header; exit 0 ;;
    *) args+=("$1") ;;
  esac
  shift
done
clean_args ${args[@]+"${args[@]}"}

# --- folder clean.sh scripts --------------------------------------------------

# folder_scripts: every clean.sh below the root that git tracks, so only
# committed (or staged) scripts run, never a copy inside node_modules/.
# .context/ is excluded by name: it holds other checkouts and throwaway trees,
# and only a local exclude file, not .gitignore, keeps it out of git.
folder_scripts() {
  git -C "$CLEAN_ROOT" ls-files --cached -- ':(glob)**/clean.sh' ':!.context' |
    grep -vx 'clean.sh' | sort -u
}

# run_folder_scripts: start every folder clean.sh at once, then print each
# report in path order. Folders are independent and the work is disk and tool
# start-up bound, so running them together is safe and much faster.
run_folder_scripts() {
  local out script i=0 failed=0 kb
  local scripts=() pids=()
  out="$(mktemp -d)"
  while IFS= read -r script; do
    [ -f "$CLEAN_ROOT/$script" ] || continue
    scripts+=("$script")
    local flag=()
    if [ "$CLEAN_APPLY" = 1 ]; then flag=(--apply); fi
    CLEAN_KB_FILE="$out/$i.kb" \
      bash "$CLEAN_ROOT/$script" ${flag[@]+"${flag[@]}"} >"$out/$i.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done < <(folder_scripts)

  # Meanwhile, this folder's own outputs.
  clean_folder
  clean_scratch

  i=0
  while [ "$i" -lt "${#pids[@]}" ]; do
    if ! wait "${pids[$i]}"; then
      heading "${scripts[$i]}"
      note "failed; its output follows"
      failed=1
    fi
    cat "$out/$i.log"
    kb="$(cat "$out/$i.kb" 2>/dev/null || echo 0)"
    CLEAN_TOTAL_KB=$((CLEAN_TOTAL_KB + kb))
    i=$((i + 1))
  done
  rm -rf "$out"
  return "$failed"
}

# --- .context/ ----------------------------------------------------------------

clean_scratch() {
  [ -d "$CLEAN_ROOT/.context" ] || return 0
  local name_args=() name dir first=1
  for name in $SCRATCH_OUTPUT_NAMES; do name_args+=(-o -name "$name"); done
  while IFS= read -r dir; do
    inside_removed_worktree "$dir" && continue
    is_scratch_output "$dir" || continue
    if [ "$first" = 1 ]; then heading ".context (scratch)"; first=0; fi
    account "$dir"
    if [ "$CLEAN_APPLY" = 1 ]; then delete_path "$dir"; fi
  done < <(
    find "$CLEAN_ROOT/.context" -name .git -prune \
      -o -type d \( "${name_args[@]:1}" \) -print -prune 2>/dev/null
  )
}

# is_scratch_output <dir>: safe to delete from .context/. Inside a checkout of
# its own (a release worktree, a smoke app with its own .git) that checkout must
# ignore it. Otherwise it sits in plain scratch, where it only has to be
# untracked here: do not ask whether .context/ is ignored, because only
# Conductor's .git/info/exclude says so.
is_scratch_output() {
  local owner
  owner="$(git -C "$(dirname "$1")" rev-parse --show-toplevel 2>/dev/null)" ||
    return 1
  if [ "$owner" != "$CLEAN_ROOT" ]; then
    is_disposable "$1"
  else
    [ -z "$(git -C "$CLEAN_ROOT" ls-files -- "$1" 2>/dev/null | head -n 1)" ]
  fi
}

# is_published <worktree>: HEAD is safe elsewhere, on a remote branch or as a
# commit of a merged PR (a squash merge deletes the only remote branch holding
# it, and CI may have pushed more commits on top, as release-prep does).
is_published() {
  local head branch merged_commits
  [ -n "$(git -C "$1" branch -r --contains HEAD 2>/dev/null)" ] && return 0
  command -v gh >/dev/null 2>&1 || return 1
  head="$(git -C "$1" rev-parse HEAD)" || return 1
  branch="$(git -C "$1" symbolic-ref --quiet --short HEAD)" || return 1
  # </dev/null: gh would otherwise drain the caller's `while read` input.
  merged_commits="$(cd "$1" && gh pr list --head "$branch" --state merged \
    --json commits --jq '.[].commits[].oid' </dev/null 2>/dev/null)" || return 1
  case $'\n'"$merged_commits"$'\n' in *$'\n'"$head"$'\n'*) return 0 ;; esac
  return 1
}

inside_removed_worktree() {
  local wt
  while IFS= read -r wt; do
    [ -n "$wt" ] || continue
    case "$1" in "$wt"/*) return 0 ;; esac
  done <<<"$REMOVED_WORKTREES"
  return 1
}

clean_worktrees() {
  [ -d "$CLEAN_ROOT/.context" ] || return 0
  heading ".context worktrees (--worktrees)"
  local gitfile wt status kb
  while IFS= read -r gitfile; do
    wt="$(dirname "$gitfile")"
    if ! status="$(git -C "$wt" status --porcelain 2>/dev/null)"; then
      skip "$wt" "git cannot read it"
      continue
    fi
    if [ -n "$status" ]; then
      skip "$wt" "uncommitted or untracked changes"
      continue
    fi
    if ! is_published "$wt"; then
      skip "$wt" "HEAD is on no remote branch and in no merged PR"
      continue
    fi
    kb="$(kb_of "$wt")"
    if [ "$CLEAN_APPLY" = 1 ] && ! git -C "$wt" worktree remove "$wt"; then
      skip "$wt" "git worktree remove refused"
      continue
    fi
    account "$wt" "$kb"
    REMOVED_WORKTREES="$REMOVED_WORKTREES$wt"$'\n'
  done < <(
    find "$CLEAN_ROOT/.context" -maxdepth 3 -name .git -type f 2>/dev/null | sort
  )
}

# --- Bazel --------------------------------------------------------------------

# The convenience symlink points at <output_base>/execroot/_main/bazel-out, so
# the output base is known without starting a Bazel server.
output_base() {
  local link="$CLEAN_ROOT/bazel-out" target
  [ -L "$link" ] || return 1
  target="$(readlink "$link")"
  echo "${target%/execroot/*}"
}

bazel_bin() {
  if [ -n "${BAZEL:-}" ]; then echo "$BAZEL"; return; fi
  command -v bazelisk || command -v bazel
}

clean_bazel_base() {
  local base bin
  base="$(output_base)" || return 0
  [ -d "$base" ] || return 0
  heading "Bazel output base (--bazel)"
  account "$base"
  [ "$CLEAN_APPLY" = 1 ] || return 0
  if bin="$(bazel_bin)"; then
    (cd "$CLEAN_ROOT" && "$bin" clean --expunge)
  else
    delete_path "$base"
  fi
}

bazel_user_root() {
  if [ -n "${BAZEL_USER_ROOT:-}" ]; then echo "$BAZEL_USER_ROOT"; return; fi
  local base
  if base="$(output_base)"; then dirname "$base"; return; fi
  case "$(uname -s)" in
    Darwin) echo "$HOME/Library/Caches/bazel/_bazel_$(id -un)" ;;
    *) echo "${XDG_CACHE_HOME:-$HOME/.cache}/bazel/_bazel_$(id -un)" ;;
  esac
}

clean_bazel_orphans() {
  local user_root marker base name workspace pid first=1
  user_root="$(bazel_user_root)"
  [ -d "$user_root" ] || return 0
  for marker in "$user_root"/*/DO_NOT_BUILD_HERE; do
    [ -f "$marker" ] || continue
    base="$(dirname "$marker")"
    name="$(basename "$base")"
    # Output bases are named by the MD5 of their workspace path.
    case "$name" in *[!0-9a-f]*) continue ;; esac
    [ "${#name}" = 32 ] || continue
    workspace="$(cat "$marker")"
    [ -n "$workspace" ] && [ ! -e "$workspace" ] || continue
    if [ "$first" = 1 ]; then
      heading "Orphaned Bazel output bases (--bazel-orphans)"
      first=0
    fi
    pid="$(cat "$base/server/server.pid.txt" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      skip "$base" "its Bazel server is still running as pid $pid"
      continue
    fi
    account "$base"
    note "was $workspace"
    if [ "$CLEAN_APPLY" = 1 ]; then delete_path "$base"; fi
  done
}

# --- main ---------------------------------------------------------------------

if [ "$CLEAN_APPLY" = 1 ]; then
  echo "Deleting from $CLEAN_ROOT"
else
  echo "Dry run for $CLEAN_ROOT"
fi
# Worktrees go first so the scratch sweep skips outputs inside the removed ones.
if [ "$WORKTREES" = 1 ]; then clean_worktrees; fi
status=0
run_folder_scripts || status=1
if [ "$BAZEL_BASE" = 1 ]; then clean_bazel_base; fi
if [ "$BAZEL_ORPHANS" = 1 ]; then clean_bazel_orphans; fi

clean_summary
unscanned=""
[ "$WORKTREES" = 1 ] || unscanned="$unscanned --worktrees"
[ "$BAZEL_BASE" = 1 ] || unscanned="$unscanned --bazel"
[ "$BAZEL_ORPHANS" = 1 ] || unscanned="$unscanned --bazel-orphans"
[ -z "$unscanned" ] || echo "Not scanned:$unscanned (see --help)."
exit "$status"
