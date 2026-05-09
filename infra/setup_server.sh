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

# --- 2. Base Dependencies ---
echo "[2/7] Installing base dependencies..."
sudo apt-get install -y -qq ffmpeg libsndfile1 build-essential curl git wget

# --- 3. UFW (Permissive SSH — hardening later) ---
echo "[3/7] Configuring Firewall (UFW)..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
# SSH abierto para TODOS por ahora (IP dinámica)
sudo ufw allow 22/tcp
# vLLM API — solo localhost por defecto
# sudo ufw allow 8000/tcp  # Descomentar si necesitas acceso remoto
sudo ufw --force enable
echo "[+] UFW active. SSH open on port 22 (global — harden later)."

# --- 4. fail2ban ---
echo "[4/7] Configuring fail2ban..."
sudo apt-get install -y -qq fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# --- 5. SSH Hardening (keys only, no passwords) ---
echo "[5/7] Hardening SSH (disable password auth)..."
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# --- 6. Docker ---
echo "[6/7] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker $USER
    rm /tmp/get-docker.sh
else
    echo "  Docker already installed: $(docker --version)"
fi

# --- 7. Data directories ---
echo "[7/7] Creating data directories..."
sudo mkdir -p /data/{input,output,workspaces,logs,models}
sudo chmod -R 777 /data

echo ""
echo "============================================"
echo " SETUP COMPLETE — $(date)"
echo "============================================"
echo " Next steps:"
echo "   1. Run: infra/deploy_vllm.sh"
echo "   2. Run: infra/deploy_pipeline.sh"
echo "============================================"
