#!/usr/bin/env bash
# Extract one version's section from CHANGELOG.md.
#
#   scripts/ci/changelog_section.sh <version> [output]   # e.g. 0.1.0

set -euo pipefail

VERSION="${1:?usage: changelog_section.sh <version> [output]}"
OUTPUT="${2:-/tmp/changelog-section.md}"

if [[ ! -f CHANGELOG.md ]]; then
    echo "::error::CHANGELOG.md is missing in $PWD. Create it, with a '## [$VERSION]' section."
    exit 1
fi

awk -v heading="## [$VERSION]" '
    index($0, heading) == 1 { found = 1; next }
    found && /^## \[/ { exit }
    found { lines[n++] = $0 }
    END {
        while (n > 0 && (lines[n - 1] ~ /^[[:space:]]*$/ || lines[n - 1] ~ /^-{3,}[[:space:]]*$/)) {
            n--
        }
        for (i = 0; i < n; i++) print lines[i]
    }
' CHANGELOG.md > "$OUTPUT"

if [[ ! -s "$OUTPUT" ]]; then
    echo "::error::CHANGELOG.md has no non-empty '## [$VERSION]' section. Rename the Unreleased heading to [$VERSION] and describe the release."
    exit 1
fi

echo "Changelog section for [$VERSION]:"
cat "$OUTPUT"
