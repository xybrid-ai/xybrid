#!/usr/bin/env bash
# Install xybrid AI skills into your project.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/tools/scripts/install-skills.sh | bash
#
# Or run directly:
#   ./install-skills.sh
#
# This adds /xybrid-init and /test-model skills to Claude Code, Codex, and other
# AI assistants that support the agents/skills/ convention.

set -euo pipefail

REPO="xybrid-ai/xybrid"
BRANCH="master"
BASE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}"

SKILLS=(
  "xybrid-init"
  "test-model"
)

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

  if command -v curl &> /dev/null; then
    curl -sSL "${url}" -o "${dest}/SKILL.md"
  elif command -v wget &> /dev/null; then
    wget -q "${url}" -O "${dest}/SKILL.md"
  else
    echo "Error: curl or wget is required" >&2
    exit 1
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
echo "  /xybrid-init    Generate model_metadata.json for any ML model"
echo "  /test-model     Test a model end-to-end with xybrid"
echo ""
echo "Quick start:"
echo "  claude /xybrid-init hexgrad/Kokoro-82M-v1.0-ONNX"
