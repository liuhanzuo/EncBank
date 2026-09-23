#!/usr/bin/env bash
set -euo pipefail
test "$(id -u)" = 0
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_CODENAME" = noble
proxy_host=$(ip route show default | awk 'NR==1 {print $3}')
proxy_url="http://${proxy_host}:7897"
install -m 0755 -d /etc/apt/keyrings
curl --proxy "$proxy_url" --connect-timeout 15 --max-time 90 -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod 0644 /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get -o Acquire::https::Proxy::download.docker.com="$proxy_url" update
DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::https::Proxy::download.docker.com="$proxy_url" install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
install -m 0755 -d /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/comem-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=$proxy_url"
Environment="HTTPS_PROXY=$proxy_url"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF
usermod -aG docker mlx
systemctl daemon-reload
systemctl enable docker.service containerd.service
systemctl restart docker.service
docker version --format '{{json .Server}}'
docker compose version
