set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../

# Sort imports
python3 -m isort avssl/ --profile black
# Autoformat code
python3 -m black avssl/ test/

set +euxo pipefail