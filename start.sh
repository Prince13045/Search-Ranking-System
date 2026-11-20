
#!/usr/bin/env bash

uvicorn app:app --host 127.0.0.1 --port 8000 &
API_PID=$!

streamlit run streamlitapp.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-10000}" \
    --server.headless true &
STREAMLIT_PID=$!

cleanup() {
    kill "$API_PID" "$STREAMLIT_PID" 2>/dev/null
}

trap cleanup EXIT INT TERM

wait -n "$API_PID" "$STREAMLIT_PID"
STATUS=$?

exit "$STATUS"
