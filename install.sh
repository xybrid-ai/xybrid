#!/bin/sh
# Xybrid CLI installer
# Usage: curl -sSL https://raw.githubusercontent.com/xybrid-ai/xybrid/master/install.sh | sh
set -e

REPO="xybrid-ai/xybrid"
BINARY_NAME="xybrid"

# --- Helpers ---

info() {
  printf "\033[1;34m==>\033[0m %s\n" "$1"
}

error() {
  printf "\033[1;31merror:\033[0m %s\n" "$1" >&2
  exit 1
}

# --- Detect platform ---

detect_platform() {
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Linux)   PLATFORM="linux" ;;
    Darwin)  PLATFORM="macos" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *)       error "Unsupported operating system: $OS" ;;
  esac

  case "$ARCH" in
    x86_64|amd64)   ARCH="x86_64" ;;
    arm64|aarch64)   ARCH="arm64" ;;
    *)               error "Unsupported architecture: $ARCH" ;;
  esac
}

# --- Detect latest release ---

get_latest_version() {
  # Can't use /releases/latest — it may return cargokit precompiled_* releases.
  # Instead, query the API for the first release whose tag starts with "v".
  API_URL="https://api.github.com/repos/$REPO/releases"

  if command -v curl >/dev/null 2>&1; then
    VERSION=$(curl -sL "$API_URL" \
      | grep '"tag_name":' \
      | sed 's/.*"tag_name": *"//;s/".*//' \
      | grep '^v' \
      | head -1)
  elif command -v wget >/dev/null 2>&1; then
    VERSION=$(wget -qO- "$API_URL" \
      | grep '"tag_name":' \
      | sed 's/.*"tag_name": *"//;s/".*//' \
      | grep '^v' \
      | head -1)
  else
    error "Neither curl nor wget found. Please install one of them."
  fi

  if [ -z "$VERSION" ]; then
    error "Could not determine latest version. Check https://github.com/$REPO/releases"
  fi
}

# --- Build artifact name ---

build_artifact_name() {
  # Release artifacts follow: xybrid-v0.1.0-beta9-macos-arm64
  ARTIFACT="${BINARY_NAME}-${VERSION}-${PLATFORM}-${ARCH}"

  # Windows binaries have .exe extension
  if [ "$PLATFORM" = "windows" ]; then
    ARTIFACT="${ARTIFACT}.exe"
  fi

  # Check if this platform/arch combo has a release binary
  case "${PLATFORM}-${ARCH}" in
    macos-arm64)    ;; # supported
    linux-x86_64)   ;; # supported
    windows-x86_64) ;; # supported
    *)
      error "No prebuilt binary for ${PLATFORM}-${ARCH}. Install from source instead:
  cargo install --git https://github.com/$REPO xybrid-cli"
      ;;
  esac
}

# --- Choose install directory ---

choose_install_dir() {
  if [ -n "$XYBRID_INSTALL_DIR" ]; then
    INSTALL_DIR="$XYBRID_INSTALL_DIR"
  elif [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
  else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
  fi
}

# --- Download and install ---

download_and_install() {
  URL="https://github.com/$REPO/releases/download/${VERSION}/${ARTIFACT}"
  TMPDIR=$(mktemp -d)
  TMPFILE="$TMPDIR/$BINARY_NAME"

  info "Downloading xybrid ${VERSION} for ${PLATFORM}-${ARCH}..."

  if command -v curl >/dev/null 2>&1; then
    HTTP_CODE=$(curl -sL -w "%{http_code}" -o "$TMPFILE" "$URL")
    if [ "$HTTP_CODE" != "200" ]; then
      rm -rf "$TMPDIR"
      error "Download failed (HTTP $HTTP_CODE). Binary may not exist for this release.
  URL: $URL
  Try installing from source: cargo install --git https://github.com/$REPO xybrid-cli"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -qO "$TMPFILE" "$URL"; then
      rm -rf "$TMPDIR"
      error "Download failed. Binary may not exist for this release.
  URL: $URL
  Try installing from source: cargo install --git https://github.com/$REPO xybrid-cli"
    fi
  fi

  chmod +x "$TMPFILE"
  mv "$TMPFILE" "$INSTALL_DIR/$BINARY_NAME"
  rm -rf "$TMPDIR"
}

# --- Verify installation ---

verify_install() {
  if ! command -v "$BINARY_NAME" >/dev/null 2>&1; then
    # Binary installed but not in PATH
    printf "\n"
    info "Installed to $INSTALL_DIR/$BINARY_NAME"
    printf "\n"
    printf "  \033[1;33mAdd to your PATH:\033[0m\n"
    printf "    export PATH=\"%s:\$PATH\"\n" "$INSTALL_DIR"
    printf "\n"
    printf "  Add the line above to your ~/.bashrc, ~/.zshrc, or equivalent.\n"
  else
    printf "\n"
    info "Installed xybrid ${VERSION} to $INSTALL_DIR/$BINARY_NAME"
    printf "\n"
    printf "  Get started:\n"
    printf "    xybrid --help\n"
    printf "    xybrid models list\n"
    printf "    xybrid run --model kokoro-82m --input-text \"Hello world\" -o output.wav\n"
    printf "\n"
  fi
}

# --- Main ---

main() {
  info "Xybrid CLI installer"
  printf "\n"

  detect_platform
  get_latest_version
  build_artifact_name
  choose_install_dir
  download_and_install
  verify_install
}

main
