#!/usr/bin/env bash
# Deploy the Pyodide static demo to S3 + invalidate CloudFront.
#
# Usage:
#   ./scripts/deploy_waveform_pyodide.sh
#   BUCKET=my-bucket DISTRIBUTION_ID=E123 ./scripts/deploy_waveform_pyodide.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src/waveform_2d/webapp/pyodide"
BUCKET="${BUCKET:-waveform2d-althaaf-2026}"
DIST_ID="${DISTRIBUTION_ID:-EPL7GNP86PSAE}"

echo "Uploading ${SRC} → s3://${BUCKET}/"
aws s3 cp "${SRC}/index.html" "s3://${BUCKET}/index.html" \
  --content-type "text/html; charset=utf-8" --cache-control "no-cache"
aws s3 cp "${SRC}/wave_engine.py" "s3://${BUCKET}/wave_engine.py" \
  --content-type "text/plain; charset=utf-8" --cache-control "no-cache"

echo "Invalidating CloudFront ${DIST_ID} …"
aws cloudfront create-invalidation --distribution-id "${DIST_ID}" --paths "/*" \
  --query 'Invalidation.{Id:Id,Status:Status}' --output json

echo "Done. Demo: https://$(aws cloudfront get-distribution --id "${DIST_ID}" \
  --query 'Distribution.DomainName' --output text 2>/dev/null || echo '<distribution-domain>')/"
