#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/chocolate-doom/chocolate-doom.git"
TAG="chocolate-doom-3.1.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REF_DIR="${ROOT_DIR}/reference"

if [ -d "${REF_DIR}/.git" ]; then
    echo "Reference repository already exists at ${REF_DIR}. Verifying tag ${TAG}..."
    CURRENT_TAG="$(git -C "${REF_DIR}" describe --tags --exact-match 2>/dev/null || true)"
    if [ "${CURRENT_TAG}" != "${TAG}" ]; then
        echo "Fetching and checking out ${TAG}..."
        git -C "${REF_DIR}" fetch --depth 1 origin tag "${TAG}"
        git -C "${REF_DIR}" checkout "${TAG}"
    else
        echo "Already at tag ${TAG}."
    fi
else
    rm -rf "${REF_DIR}"
    echo "Cloning ${REPO_URL} at tag ${TAG} into ${REF_DIR}..."
    git clone --depth 1 --branch "${TAG}" -c advice.detachedHead=false "${REPO_URL}" "${REF_DIR}"
fi

EXPECTED_FILE="${REF_DIR}/src/hexen/p_map.c"
if [ ! -f "${EXPECTED_FILE}" ]; then
    echo "Error: Expected reference file not found: ${EXPECTED_FILE}" >&2
    exit 1
fi
echo "Reference source ready at ${REF_DIR} (${TAG})."
