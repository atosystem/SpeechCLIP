set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../

# Check imports
isort -c avssl/
# Check code format
black avssl/ test/ --check --experimental-string-processing
# Check lint: Not checking the code in website

set +euxo pipefail
