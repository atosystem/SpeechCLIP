set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../

# Check imports
python3 -m isort -c avssl/ --profile black
# Check code format
python3 -m black avssl/ test/
# Check lint: Not checking the code in website

set +euxo pipefail