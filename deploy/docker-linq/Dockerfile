FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openvpn iproute2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY connect-linq.sh /usr/local/sbin/connect-linq
COPY tcp_forward.py /usr/local/lib/linq/tcp_forward.py
RUN chmod 0755 /usr/local/sbin/connect-linq

EXPOSE 18184/tcp
HEALTHCHECK --interval=2s --timeout=3s --start-period=90s --retries=2 \
    CMD ["python3", "/usr/local/lib/linq/tcp_forward.py", "--health"]

CMD ["sleep", "infinity"]
