#!/usr/bin/env bash
# Stamp the workspace version from a release tag.
#
#   scripts/ci/stamp_version.sh <version>       # e.g. 0.1.0, 0.1.0-beta.1

set -euo pipefail

VERSION="${1:?usage: stamp_version.sh <version>}"

CURRENT=$(cargo metadata --no-deps --format-version=1 \
    | jq -r '.packages[] | select(.name == "hydr") | .version')

if [[ "$VERSION" == "$CURRENT" ]]; then
    echo "Version already $VERSION, nothing to stamp."
    exit 0
fi

perl -i -pe 'if (!$done && /^version = "/) { s/^version = ".*"/version = "'"$VERSION"'"/; $done=1 }' Cargo.toml
perl -i -pe 's/(hydr-[a-z]+ = \{ path = "crates\/hydr-[a-z]+", version = ")='"$CURRENT"'(" \})/$1='"$VERSION"'$2/g' Cargo.toml

echo "Stamped hydr workspace: $CURRENT -> $VERSION"
