#!/bin/sh
set -eu

PROFILE=/tmp/client.ovpn
AUTH=/run/secrets/vpn-auth
LOG=/run/secrets/openvpn.log
READY=/run/secrets/ready
VPN_PID=/run/secrets/vpn.pid
PROXY_PID=/run/secrets/proxy.pid

umask 077
mkdir -p /run/secrets
if [ ! -r "$PROFILE" ] || [ ! -s "$AUTH" ]; then
    printf 'vpn_input_missing\n'
    exit 1
fi

vpn_pid=''
proxy_pid=''
keep_running=0
cleanup_failed_start() {
    if [ "$keep_running" -eq 0 ]; then
        rm -f "$READY" "$VPN_PID" "$PROXY_PID"
        if [ -n "$vpn_pid" ] && kill -0 "$vpn_pid" 2>/dev/null; then kill "$vpn_pid" 2>/dev/null || true; fi
        if [ -n "$proxy_pid" ] && kill -0 "$proxy_pid" 2>/dev/null; then kill "$proxy_pid" 2>/dev/null || true; fi
    fi
}
trap cleanup_failed_start EXIT INT TERM

openvpn --config "$PROFILE" --auth-user-pass "$AUTH" \
    --route 10.0.113.1 255.255.255.255 --daemon --log "$LOG" --writepid "$VPN_PID"

attempt=0
while [ "$attempt" -lt 60 ]; do
    if [ -s "$VPN_PID" ]; then
        vpn_pid=$(cat "$VPN_PID")
        case "$vpn_pid" in
            ''|*[!0-9]*) vpn_pid='' ;;
        esac
    fi
    if [ -n "$vpn_pid" ] && kill -0 "$vpn_pid" 2>/dev/null && \
       grep -q 'Initialization Sequence Completed' "$LOG" 2>/dev/null; then
        break
    fi
    if { [ -n "$vpn_pid" ] && ! kill -0 "$vpn_pid" 2>/dev/null; } || \
       grep -Eqi 'AUTH_FAILED|Options error|Exiting due to fatal error' "$LOG" 2>/dev/null; then
        printf 'vpn_connection_failed\n'
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done
if [ -z "$vpn_pid" ] || ! kill -0 "$vpn_pid" 2>/dev/null || \
   ! grep -q 'Initialization Sequence Completed' "$LOG" 2>/dev/null; then
    printf 'vpn_connection_timeout\n'
    exit 1
fi

python3 /tmp/tcp_forward.py >/dev/null 2>&1 &
proxy_pid=$!
attempt=0
remote_ready=0
while [ "$attempt" -lt 8 ]; do
    if python3 /tmp/tcp_forward.py --check >/dev/null 2>&1; then
        remote_ready=1
        break
    fi
    if [ -n "$vpn_pid" ] && ! kill -0 "$vpn_pid" 2>/dev/null; then
        printf 'vpn_connection_failed\n'
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 1
done
if [ "$remote_ready" -ne 1 ]; then
    printf 'vpn_target_unreachable\n'
    exit 1
fi

printf '%s\n' "$proxy_pid" > "$PROXY_PID"
touch "$READY"
keep_running=1
trap - EXIT INT TERM
printf 'vpn_connected_forward_ready\n'
