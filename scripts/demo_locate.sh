#!/usr/bin/env bash
# Demo of the locate API: shortest playable segment reaching a fraction of
# the video-wide positive contribution, with a focus on how NEGATIVE
# contributions change which segment is chosen.
#
# Prereq: service listening on ${BASE:-http://localhost:8000}
#   docker compose up --build        # or: uvicorn app.main:app --port 8000
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"

req() { # method path [json-body] -> body + status line
  local method="$1" path="$2" body="${3:-}"
  echo
  echo "### $method $path ${body:+ $body}"
  if [ -n "$body" ]; then
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path" \
      -H 'Content-Type: application/json' --data "$body"
  else
    curl -s -w $'\nHTTP %{http_code}\n' -X "$method" "$BASE$path"
  fi
}

locate() { # ratio -> locate request body (query = unit vector on dimension 0)
  printf '{"query":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"ratio":%s}' "$1"
}

# Eight shots, all with equal attention score (softmax -> equal weights).
# Only dimension 0 varies so the query [1,0,...,0] reads it directly.
# Dot products (feature[0]):  +0.9  -0.9  +0.9  +0.9 | +0.9  -0.9  +0.9  +0.9
# Shots 3 and 7 are hidden later to show playable-run splitting.
SHOTS='{"shots":[
 {"feature":[ 0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[-0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[ 0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[ 0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[ 0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[-0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[ 0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0},
 {"feature":[ 0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"score":0}
]}'

echo "=== 1. import 8 shots (equal softmax weights 1/8) ==="
curl -s -w $'\nHTTP %{http_code}\n' -X POST "$BASE/videos/loc-demo/shots" \
  -H 'Content-Type: application/json' --data "$SHOTS"

# Each shot weight is 1/8 = 0.125, so per-shot contribution magnitude is
# 0.125 * 0.9 = 0.1125. Six positive shots -> positive_total = 0.675.
echo
echo "=== 2. locate ratio=0.1 (target 0.0675): any single positive shot (0.1125) qualifies ==="
echo "    expect the earliest length-1 segment [0,1)"
req POST /videos/loc-demo/locate "$(locate 0.1)"

echo
echo "=== 3. locate ratio=0.5 (target 0.3375) ==="
echo "    A length-3 window starting at 0 sums 0.1125-0.1125+0.1125 = 0.1125 (the"
echo "    negative shot cancels one positive). The shortest qualifying window is the"
echo "    all-positive run [2,5): length 3, sum 0.3375 -- NEGATIVE shots force a longer,"
echo "    differently placed segment than a positives-only scan would give."
req POST /videos/loc-demo/locate "$(locate 0.5)"

echo
echo "=== 4. locate ratio=1.0 (target = positive_total = 0.675) ==="
echo "    Reaching 100% requires including every positive shot, which also drags in"
echo "    both negative shots, netting only 0.45 < 0.675 -> no_qualifying_interval."
req POST /videos/loc-demo/locate "$(locate 1.0)"

echo
echo "=== 5. hide shots [3,5) and [7,8): playable runs split, segments cannot cross them ==="
req POST /videos/loc-demo/restrictions '{"order_id":"wall-a","start":3,"end":5}'
req POST /videos/loc-demo/restrictions '{"order_id":"wall-b","start":7,"end":8}'
echo
echo "    ratio=0.5 again: the largest playable run is [0,3) whose net is only 0.1125,"
echo "    so no contiguous playable segment reaches half the positive total."
req POST /videos/loc-demo/locate "$(locate 0.5)"

echo
echo "=== 6. revoke the walls: the same locate call works again ==="
req DELETE /videos/loc-demo/restrictions/wall-a
req DELETE /videos/loc-demo/restrictions/wall-b
req POST /videos/loc-demo/locate "$(locate 0.5)"

echo
echo "=== 7. no positive contribution: flip every feature against the query ==="
curl -s -w $'\nHTTP %{http_code}\n' -X POST "$BASE/videos/loc-neg/shots" \
  -H 'Content-Type: application/json' \
  --data '{"shots":[{"feature":[-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5],"score":0}]}'
req POST /videos/loc-neg/locate "$(locate 0.5)"
