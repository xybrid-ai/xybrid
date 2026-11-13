#!/usr/bin/env bash
# Install xybrid AI skills into your project.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/tools/scripts/install-skills.sh | bash
#
# Or run directly:
#   ./install-skills.sh
#
# This adds /xybrid-init, /test-model and /xybrid-connect skills to Claude Code,
# Codex, and other AI assistants that support the agents/skills/ convention.
#
# xybrid-connect also ships a reference script, used where the xybrid CLI
# can't be installed. It is downloaded next to the skill and checked against
# the SHA-256 that the skill's SKILL.md prints.

set -euo pipefail

REPO="xybrid-ai/xybrid"
BRANCH="master"
BASE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}"

SKILLS=(
  "xybrid-init"
  "test-model"
  "xybrid-connect"
)

# xybrid-connect's reference script, relative to the skill's directory.
CONNECT_SCRIPT="scripts/xybrid_agent_connect.py"

# Download one file. An HTTP error fails instead of saving the error page.
fetch() {
  if command -v curl &> /dev/null; then
    curl -fsSL "$1" -o "$2"
  elif command -v wget &> /dev/null; then
    wget -q "$1" -O "$2"
  else
    echo "Error: curl or wget is required" >&2
    exit 1
  fi
}

# Print a file's SHA-256, or nothing when no tool for it is installed.
sha256_of() {
  if command -v sha256sum &> /dev/null; then
    sha256sum "$1" | cut -d ' ' -f 1
  elif command -v shasum &> /dev/null; then
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

echo "Installing xybrid AI skills..."
echo ""

# Create canonical location
mkdir -p agents/skills

for skill in "${SKILLS[@]}"; do
  url="${BASE_URL}/agents/skills/${skill}/SKILL.md"
  dest="agents/skills/${skill}"

  mkdir -p "${dest}"

  if [ -f "${dest}/SKILL.md" ]; then
    echo "  Updating ${skill}..."
  else
    echo "  Installing ${skill}..."
  fi

  fetch "${url}" "${dest}/SKILL.md"

  if [ "${skill}" = "xybrid-connect" ]; then
    script="${dest}/${CONNECT_SCRIPT}"
    mkdir -p "$(dirname "${script}")"
    fetch "${BASE_URL}/${script}" "${script}"

    expected="$(sed -n 's/^# sha256: \([0-9a-f]\{64\}\)$/\1/p' "${dest}/SKILL.md" | head -n 1)"
    actual="$(sha256_of "${script}")"
    if [ -z "${actual}" ]; then
      echo "  Warning: no sha256sum or shasum found, so ${script} was not checked"
    elif [ "${actual}" != "${expected}" ]; then
      rm -f "${script}"
      echo "Error: ${script} does not match the SHA-256 in ${dest}/SKILL.md." >&2
      echo "The script was removed. Run this installer again in a few minutes." >&2
      exit 1
    fi
  fi
done

# Set up symlinks for AI tools that use their own directories
for tool_dir in .claude .codex; do
  skills_link="${tool_dir}/skills"
  if [ -L "${skills_link}" ]; then
    echo "  ${skills_link} symlink already exists"
  elif [ -d "${skills_link}" ]; then
    echo "  Warning: ${skills_link} is a directory, skipping symlink"
  else
    mkdir -p "${tool_dir}"
    ln -s ../agents/skills "${skills_link}"
    echo "  Created ${skills_link} -> agents/skills/"
  fi
done

echo ""
echo "Done! Skills installed to agents/skills/"
echo ""
echo "Available skills:"
echo "  /xybrid-init     Generate model_metadata.json for any ML model"
echo "  /test-model      Test a model end-to-end with xybrid"
echo "  /xybrid-connect  Connect this project to the Xybrid console, no sign-up (xybrid CLI 0.18.0+)"
echo ""
echo "Quick start:"
echo "  claude /xybrid-init hexgrad/Kokoro-82M-v1.0-ONNX"
echo "  claude /xybrid-connect"
