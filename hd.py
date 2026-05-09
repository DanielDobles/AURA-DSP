"""
╔══════════════════════════════════════════════════════════════╗
║  AARS HQ — Headquarters Control Script                      ║
║  Control everything from your Windows terminal.              ║
║  Usage: python hd.py <command>                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────
SERVER_IP = "129.212.179.98"
SERVER_USER = "root"
SSH_KEY = os.path.expanduser("~/.ssh/id_gpu_droplet")
SSH_BASE = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "{SSH_KEY}" {SERVER_USER}@{SERVER_IP}'
SCP_BASE = f'scp -o StrictHostKeyChecking=no -i "{SSH_KEY}"'

PROJECT_ROOT = Path(__file__).parent
INFRA_DIR = PROJECT_ROOT / "infra"
BRIDGE_DIR = PROJECT_ROOT / "bridge"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"


# ─── Helpers ───────────────────────────────────────────────
def ssh(cmd: str, stream: bool = False) -> int:
    """Execute a command on the remote server via SSH."""
    full_cmd = f'{SSH_BASE} "{cmd}"'
    if stream:
        return subprocess.call(full_cmd, shell=True)
    else:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode


def scp_upload(local: str, remote: str) -> int:
    """Upload a file/dir to the server."""
    cmd = f'{SCP_BASE} -r "{local}" {SERVER_USER}@{SERVER_IP}:"{remote}"'
    return subprocess.call(cmd, shell=True)


def scp_download(remote: str, local: str) -> int:
    """Download a file/dir from the server."""
    cmd = f'{SCP_BASE} -r {SERVER_USER}@{SERVER_IP}:"{remote}" "{local}"'
    return subprocess.call(cmd, shell=True)


def banner(text: str):
    w = max(len(text) + 4, 50)
    print(f"\n{'═' * w}")
    print(f"  {text}")
    print(f"{'═' * w}\n")


# ─── Commands ──────────────────────────────────────────────

def cmd_status(args):
    """Check server status: SSH, Docker, GPU, vLLM."""
    banner("SERVER STATUS")

    print("▶ SSH Connection...")
    rc = ssh("echo 'SSH OK'")
    if rc != 0:
        print("❌ Cannot reach server. Is the Droplet ON?")
        return

    print("\n▶ System Uptime...")
    ssh("uptime")

    print("\n▶ GPU (rocm-smi)...")
    ssh("rocm-smi --showuse --showmemuse 2>/dev/null || echo 'rocm-smi not available'")

    print("\n▶ Docker Containers...")
    ssh("docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || echo 'Docker not running'")

    print("\n▶ vLLM Qwen3 (brain) — port 8000...")
    ssh("curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo 'Qwen3 not responding'")

    print("\n▶ vLLM Qwen2-Audio (listener) — port 8001...")
    ssh("curl -s http://localhost:8001/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo 'Qwen2-Audio not responding'")

    print("\n▶ Disk Space...")
    ssh("df -h / /data 2>/dev/null || df -h /")

    print("\n▶ VRAM Usage...")
    ssh("rocm-smi --showmeminfo vram 2>/dev/null || echo 'N/A'")


def cmd_deploy(args):
    """Full deployment: upload scripts, harden, deploy vLLM."""
    banner("FULL DEPLOYMENT")

    # Step 1: Upload infra scripts
    print("▶ Uploading infrastructure scripts...")
    ssh("mkdir -p /tmp/aars/infra")
    for script in INFRA_DIR.glob("*.sh"):
        print(f"  → {script.name}")
        scp_upload(str(script), f"/tmp/aars/infra/{script.name}")

    # Step 2: Make executable
    ssh("chmod +x /tmp/aars/infra/*.sh")

    # Step 3: Run setup
    print("\n▶ Running server setup...")
    ssh("bash /tmp/aars/infra/setup_server.sh", stream=True)

    # Step 4: Deploy vLLM
    print("\n▶ Deploying vLLM (Qwen3 + Qwen2-Audio)...")
    ssh("bash /tmp/aars/infra/deploy_vllm.sh", stream=True)


def cmd_setup(args):
    """Server hardening only (no vLLM)."""
    banner("SERVER SETUP ONLY")
    ssh("mkdir -p /tmp/aars/infra")
    scp_upload(str(INFRA_DIR / "setup_server.sh"), "/tmp/aars/infra/setup_server.sh")
    ssh("chmod +x /tmp/aars/infra/setup_server.sh && bash /tmp/aars/infra/setup_server.sh", stream=True)


def cmd_vllm(args):
    """Deploy/restart vLLM only."""
    banner("vLLM DEPLOYMENT")
    ssh("mkdir -p /tmp/aars/infra")
    scp_upload(str(INFRA_DIR / "deploy_vllm.sh"), "/tmp/aars/infra/deploy_vllm.sh")
    ssh("chmod +x /tmp/aars/infra/deploy_vllm.sh && bash /tmp/aars/infra/deploy_vllm.sh", stream=True)


def cmd_gpu(args):
    """Show GPU status (rocm-smi)."""
    banner("GPU STATUS")
    ssh("rocm-smi", stream=True)


def cmd_logs(args):
    """Show vLLM container logs."""
    banner("vLLM LOGS")
    lines = args.lines if hasattr(args, "lines") else 50
    print("── Qwen3 (brain) ──")
    ssh(f"docker logs --tail {lines} vllm-qwen3 2>&1", stream=True)
    print("\n── Qwen2-Audio (listener) ──")
    ssh(f"docker logs --tail {lines} vllm-qwen2-audio 2>&1", stream=True)


def cmd_bridge(args):
    """Start the Bridge (local file sync)."""
    banner("STARTING BRIDGE")
    print("Launching bridge.py...")
    subprocess.call(
        [sys.executable, str(BRIDGE_DIR / "bridge.py")],
        cwd=str(BRIDGE_DIR),
    )


def cmd_upload(args):
    """Upload specific files to /data/input on server."""
    banner("MANUAL UPLOAD")
    files = args.files
    for f in files:
        if not os.path.exists(f):
            print(f"❌ File not found: {f}")
            continue
        fname = os.path.basename(f)
        print(f"▶ Uploading {fname}...")
        rc = scp_upload(f, f"/data/input/{fname}")
        if rc == 0:
            # Verify hash
            local_md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
            print(f"  Local MD5: {local_md5}")
            ssh(f"md5sum /data/input/{fname}")
            print(f"  ✅ {fname} uploaded")
        else:
            print(f"  ❌ Upload failed for {fname}")


def cmd_download(args):
    """Download completed files from /data/output."""
    banner("DOWNLOAD COMPLETED FILES")
    local_out = args.dest or str(PROJECT_ROOT / "output")
    os.makedirs(local_out, exist_ok=True)

    # List remote files
    result = subprocess.run(
        f'{SSH_BASE} "ls /data/output/ 2>/dev/null"',
        shell=True, capture_output=True, text=True
    )
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

    if not files:
        print("No completed files on server.")
        return

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  ▶ Downloading {f}...")
        scp_download(f"/data/output/{f}", os.path.join(local_out, f))
    print(f"\n✅ Files saved to: {local_out}")


def cmd_pipeline(args):
    """Upload and build the pipeline container on server."""
    banner("DEPLOY PIPELINE")

    print("▶ Uploading pipeline code...")
    ssh("mkdir -p /opt/aars/pipeline")
    scp_upload(str(PIPELINE_DIR), "/opt/aars/")

    print("\n▶ Building Docker image...")
    ssh("docker build -t aars-pipeline /opt/aars/pipeline/", stream=True)


def cmd_run(args):
    """Run the pipeline (process all files in /data/input)."""
    banner("RUN PIPELINE")
    ssh("docker run --rm --device /dev/kfd --device /dev/dri --group-add video "
        "-v /data:/data "
        "-e VLLM_BASE_URL=http://localhost:8000/v1 "
        "-e VLLM_AUDIO_URL=http://localhost:8001/v1 "
        "-e HSA_OVERRIDE_GFX_VERSION=9.4.2 "
        "--network host "
        "aars-pipeline", stream=True)


def cmd_shell(args):
    """Open an interactive SSH shell to the server."""
    banner("INTERACTIVE SHELL")
    os.system(f'ssh -i "{SSH_KEY}" {SERVER_USER}@{SERVER_IP}')


def cmd_stop(args):
    """Stop all containers on the server."""
    banner("STOPPING ALL CONTAINERS")
    ssh("docker stop $(docker ps -q) 2>/dev/null || echo 'No containers running'")
    print("✅ All containers stopped.")


def cmd_cost(args):
    """Estimate session cost based on uptime."""
    banner("COST ESTIMATE")
    result = subprocess.run(
        f'{SSH_BASE} "cat /proc/uptime"',
        shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ Cannot reach server.")
        return
    uptime_seconds = float(result.stdout.strip().split()[0])
    hours = uptime_seconds / 3600
    cost = hours * 1.99
    print(f"  Uptime:  {hours:.2f} hours")
    print(f"  Rate:    $1.99/hr")
    print(f"  Cost:    ${cost:.2f}")
    print(f"\n  ⚠️  Don't forget to power off when done!")


# ─── CLI Parser ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="hd.py",
        description="AARS Headquarters — Control the audio restoration swarm from your terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hd.py status          Check server health
  python hd.py deploy          Full deployment (setup + vLLM)
  python hd.py gpu             GPU status
  python hd.py bridge          Start local file sync
  python hd.py upload f1.wav   Upload files manually
  python hd.py run             Run the pipeline
  python hd.py cost            Check how much you've spent
  python hd.py shell           SSH into the server
  python hd.py stop            Stop all containers
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Status
    sub.add_parser("status", help="Check server status")

    # Deploy
    sub.add_parser("deploy", help="Full deployment (setup + vLLM)")
    sub.add_parser("setup", help="Server hardening only")
    sub.add_parser("vllm", help="Deploy/restart vLLM only")
    sub.add_parser("pipeline", help="Build pipeline container")

    # Operations
    sub.add_parser("run", help="Run the audio restoration pipeline")
    sub.add_parser("bridge", help="Start local file sync daemon")

    # Files
    p_upload = sub.add_parser("upload", help="Upload files to server")
    p_upload.add_argument("files", nargs="+", help="Files to upload")

    p_download = sub.add_parser("download", help="Download completed files")
    p_download.add_argument("--dest", default=None, help="Local destination folder")

    # Monitoring
    sub.add_parser("gpu", help="Show GPU status (rocm-smi)")

    p_logs = sub.add_parser("logs", help="Show vLLM logs")
    p_logs.add_argument("--lines", type=int, default=50, help="Number of log lines")

    sub.add_parser("cost", help="Estimate session cost")

    # Admin
    sub.add_parser("shell", help="Interactive SSH shell")
    sub.add_parser("stop", help="Stop all Docker containers")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "status": cmd_status,
        "deploy": cmd_deploy,
        "setup": cmd_setup,
        "vllm": cmd_vllm,
        "pipeline": cmd_pipeline,
        "run": cmd_run,
        "bridge": cmd_bridge,
        "upload": cmd_upload,
        "download": cmd_download,
        "gpu": cmd_gpu,
        "logs": cmd_logs,
        "cost": cmd_cost,
        "shell": cmd_shell,
        "stop": cmd_stop,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
