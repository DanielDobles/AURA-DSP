#!/bin/bash
# AARS — Server Setup (ROCm MI300X)
# Target: Ubuntu 24.04
# NOTA: NO restringir SSH por IP aquí. Tu IP residencial es DINÁMICA.
# El hardening de IP se hace DESPUÉS de confirmar conectividad estable.

set -e

echo "============================================"
echo " AARS Server Setup — $(date)"
echo "============================================"

# --- 1. System Update ---
echo "[1/7] Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# --- 2. Data directories (Early for persistence) ---
echo "[2/7] Creating data directories..."
sudo mkdir -p /data/{input,output,workspaces,logs,models}
sudo chmod -R 777 /data

# --- 3. Base Dependencies ---
echo "[3/7] Installing base dependencies..."
sudo apt-get install -y -qq ffmpeg libsndfile1 build-essential curl git wget

# --- 4. UFW (Permissive SSH — hardening later) ---
echo "[4/7] Configuring Firewall (UFW)..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
# SSH abierto para TODOS por ahora (IP dinámica)
# Usar insert 1 para asegurar que esté por encima de cualquier LIMIT rule
sudo ufw insert 1 allow 22/tcp
sudo ufw --force enable
echo "[+] UFW active. SSH open on port 22."

# --- 5. fail2ban ---
echo "[5/7] Configuring fail2ban..."
sudo apt-get install -y -qq fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# --- 6. SSH Hardening (keys only, no passwords) ---
echo "[6/7] Hardening SSH (disable password auth)..."
sudo sed -i 's/^[#]*PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^[#]*PasswordAuthentication no/PasswordAuthentication no/' /etc/ssh/sshd_config
# Reload instead of restart to avoid dropping current connection
sudo systemctl reload ssh

# --- 7. Docker ---
echo "[7/7] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker $USER
    rm /tmp/get-docker.sh
else
    echo "  Docker already installed: $(docker --version)"
fi

echo ""
echo "============================================"
echo " SETUP COMPLETE — $(date)"
echo "============================================"
echo " Next steps:"
echo "   1. Run: infra/deploy_vllm.sh"
echo "   2. Run: infra/deploy_pipeline.sh"
echo "============================================"
