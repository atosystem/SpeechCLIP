set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../

# Sort imports
isort avssl/
# Autoformat code
black avssl/ test/

set +euxo pipefail