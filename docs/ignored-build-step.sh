#!/bin/sh
# Vercel's Ignored Build Step for the docs site: vercel.json runs it from docs/.
# Exit 0 skips the build and exit 1 builds. Any other status fails the
# deployment, so every error path below exits 1 instead.
#
# The site builds when a file it is built from changed since the last
# successful deployment of the branch. development/, sdk/, guides/,
# telemetry/, assets/ and the Markdown files at this level are repository
# docs that the site never reads, so changing only them skips the build.

base="${VERCEL_GIT_PREVIOUS_SHA:-HEAD^}"

# The base is missing after a force-push, or when the clone does not reach
# back to it. Build rather than guess.
git cat-file -e "$base^{commit}" 2>/dev/null || exit 1

if git diff --quiet "$base" HEAD -- . \
  ':(exclude)development' ':(exclude)sdk' ':(exclude)guides' \
  ':(exclude)telemetry' ':(exclude)assets' ':(exclude,glob)*.md'; then
  exit 0
fi
exit 1
