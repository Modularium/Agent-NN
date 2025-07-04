#!/usr/bin/env sh
set -eu

SCRIPTS="scripts/deploy/build_frontend.sh scripts/deploy/start_services.sh scripts/deploy/dev_reset.sh scripts/setup.sh"
for s in $SCRIPTS; do
    echo "Testing $s"
    ./$s --help >/dev/null
done

echo "All script help commands succeeded"
