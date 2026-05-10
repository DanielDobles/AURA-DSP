"""
╔══════════════════════════════════════════════════════════════╗
║  AARS HQ — The Hub (v3 Autonomous Orchestrator)              ║
║  One-command deployment and restoration control.              ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.align import Align
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    print("Error: 'rich' library not found. Run: pip install rich")
    sys.exit(1)

console = Console()

# ─── Configuration ─────────────────────────────────────────
SERVER_IP = "129.212.179.98"
SERVER_USER = "root"
SSH_KEY = os.path.expanduser("~/.ssh/id_gpu_droplet")

SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH_BASE = f'ssh {SSH_OPTS} -i "{SSH_KEY}" {SERVER_USER}@{SERVER_IP}'
SCP_BASE = f'scp {SSH_OPTS} -i "{SSH_KEY}"'

PROJECT_ROOT = Path(__file__).parent
INFRA_DIR = PROJECT_ROOT / "infra"
BRIDGE_DIR = PROJECT_ROOT / "bridge"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

RATE_PER_HOUR = 1.99

# ─── UI Visuals ────────────────────────────────────────────
BANNER = """
[bold white]  █████╗  █████╗ ██████╗ ███████╗[/bold white]
[bold cyan] ██╔══██╗██╔══██╗██╔══██╗██╔════╝[/bold cyan]
[bold magenta] ███████║███████║██████╔╝███████╗[/bold magenta]
[bold yellow] ██╔══██║██╔══██║██╔══██╗╚════██║[/bold yellow]
[bold white] ██║  ██║██║  ██║██║  ██║███████║[/bold white]
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
    [bold dim]       A U T O N O M O U S   H U B       [/bold dim]
"""

METER_COLUMN = TextColumn("[progress.percentage]{task.percentage:>3.0f}%")

# ─── Core Logic ────────────────────────────────────────────

class AARSHub:
    def __init__(self):
        self.active_step = ""
        self.error_count = 0

    def log_phase(self, num: int, text: str):
        console.print(f"\n[bold cyan]▶ PHASE {num}:[/bold cyan] [white]{text}[/white]")

    def exec_local(self, cmd: str) -> bool:
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            return True
        except Exception as e:
            console.print(f"[red]Local Error: {e}[/red]")
            return False

    def exec_remote(self, cmd: str, capture: bool = False, retries: int = 2):
        full_cmd = f'{SSH_BASE} "{cmd}"'
        for i in range(retries + 1):
            try:
                if capture:
                    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, check=True)
                    return result.stdout.strip()
                else:
                    # Stream output for better UX
                    return self.exec_remote_stream(full_cmd)
            except subprocess.CalledProcessError as e:
                if i < retries and e.returncode == 255:
                    console.print(f"[dim]  (SSH retry {i+1}/{retries}...)[/]")
                    time.sleep(2)
                    continue
                err_msg = e.stderr if e.stderr else str(e)
                if "Connection timed out" in err_msg or "Permission denied" in err_msg:
                    self.handle_lockout()
                raise e

    def exec_remote_stream(self, full_cmd: str) -> bool:
        """Runs a remote command and streams output to console."""
        process = subprocess.Popen(
            full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        )
        for line in iter(process.stdout.readline, ""):
            if line:
                console.print(f"[dim]  remote:[/][white] {line.strip()}[/]")
        
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, full_cmd)
        return True

    def handle_lockout(self):
        console.print(Panel.fit(
            "[bold red]CRITICAL: CONNECTION FAILED[/bold red]\n\n"
            "Possible causes:\n"
            "1. Droplet is [bold yellow]OFF[/bold yellow].\n"
            "2. Your [bold yellow]IP changed[/bold yellow] (Dynamic IP).\n\n"
            "Action: Go to Cloud Console and run:\n"
            "[bold cyan]ufw allow 22/tcp[/bold cyan]",
            title="Lockout Detected", border_style="red"
        ))
        sys.exit(1)

    def ignite(self):
        """The 'One-Click' Orchestrator"""
        console.clear()
        console.print(Align.center(BANNER))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # --- PHASE 1: Pre-flight ---
            task1 = progress.add_task("[cyan]PHASE 1: Local Pre-flight...", total=3)
            
            # 1.1 Check SSH Key
            if not os.path.exists(SSH_KEY):
                console.print(f"[red]Error: SSH Key not found at {SSH_KEY}[/red]")
                sys.exit(1)
            progress.advance(task1)
            
            # 1.2 Check Directories
            for d in [INFRA_DIR, BRIDGE_DIR, PIPELINE_DIR]:
                if not d.exists():
                    console.print(f"[red]Error: Missing directory {d.name}[/red]")
                    sys.exit(1)
            progress.advance(task1)
            
            # 1.3 Test Remote Connection
            try:
                self.exec_remote("echo 1")
            except:
                self.handle_lockout()
            progress.advance(task1)
            
            # --- PHASE 2: Infrastructure ---
            task2 = progress.add_task("[magenta]PHASE 2: Hardening MI300X...", total=3)
            
            # 2.1 Upload scripts (Single connection)
            progress.update(task2, description="[magenta]Uploading Infrastructure Scripts...")
            self.exec_remote("mkdir -p /tmp/aars")
            scp_upload(str(INFRA_DIR), "/tmp/aars/")
            progress.advance(task2)
            
            # 2.2 Run Setup
            progress.update(task2, description="[magenta]Installing Docker & UFW (Hold tight)...")
            try:
                self.exec_remote("chmod +x /tmp/aars/infra/*.sh && bash /tmp/aars/infra/setup_server.sh")
            except subprocess.CalledProcessError as e:
                if e.returncode == 255:
                    console.print("[yellow]Notice: Connection dropped during SSH reload. This is expected. Re-verifying...[/yellow]")
                    time.sleep(2)
                    self.exec_remote("echo 1") # Verify re-connection
                else:
                    raise e
            progress.advance(task2)
            
            # --- PHASE 3: vLLM Swarm ---
            task3 = progress.add_task("[yellow]PHASE 3: Deploying Qwen Swarm...", total=3)
            
            # Check if models are ALREADY running to save time
            progress.update(task3, description="[yellow]Checking for existing Swarm models...")
            q3_online = False
            try:
                res = self.exec_remote("curl -s http://localhost:8000/v1/models | grep -q Qwen && echo 1 || echo 0", capture=True)
                q3_online = "1" in res
            except: pass
            
            qa_online = False
            try:
                res = self.exec_remote("curl -s http://localhost:8001/v1/models | grep -q Qwen && echo 1 || echo 0", capture=True)
                qa_online = "1" in res
            except: pass
            
            if q3_online and qa_online:
                console.print("[green]  [✓] vLLM models already online. Skipping redeploy.[/]")
                progress.advance(task3, 2)
            else:
                # 3.0 Purge ALL old containers (AURYGA leftovers + previous runs)
                progress.update(task3, description="[yellow]Purging old containers (AURYGA cleanup)...")
                self.exec_remote("docker rm -f $(docker ps -aq) 2>/dev/null || true")
                progress.advance(task3)
                
                # 3.1 Deploy fresh
                progress.update(task3, description="[yellow]Initializing Qwen3 Brain & Qwen2-Audio Hearing...")
                try:
                    self.exec_remote("bash /tmp/aars/infra/deploy_vllm.sh")
                except subprocess.CalledProcessError as e:
                    err = e.stderr if e.stderr else str(e)
                    console.print(Panel(
                        f"[bold red]vLLM DEPLOY FAILED[/bold red]\n\n"
                        f"[white]{err[:500]}[/white]\n\n"
                        "[dim]Try: python hd.py logs[/dim]",
                        border_style="red", title="Error Detail"
                    ))
                    sys.exit(1)
                progress.advance(task3)
            
            self.ignite_phase_4()

        console.print(Panel(
            "[bold green]SYSTEM FULLY IGNITED[/bold green]\n\n"
            "The AARS Swarm is online on the MI300X.\n"
            "Qwen3 is reasoning. Qwen2-Audio is listening.\n\n"
            "Next: Drop files into [bold cyan]C:\\Suno_Restoration\\Input[/bold cyan]",
            border_style="green", title="AARS Core"
        ))
        
        # Auto-start Bridge
        self.start_bridge()

    def start_bridge(self):
        console.print("\n[bold cyan]📡 ACTIVATING BRIDGE SYNC...[/bold cyan]")
        bridge_script = BRIDGE_DIR / "bridge.py"
        if bridge_script.exists():
            subprocess.call([sys.executable, str(bridge_script)], cwd=str(BRIDGE_DIR))
        else:
            console.print("[yellow]Bridge script not found, skipping auto-sync.[/yellow]")

    def ignite_phase_4(self):
        """Phase 4: Build Restoration Container"""
        self.exec_remote("mkdir -p /opt/aars/pipeline")
        scp_upload(str(PIPELINE_DIR), "/opt/aars/")
        self.exec_remote("docker build -t aars-pipeline /opt/aars/pipeline/")

# ─── Standard CLI Mapping ──────────────────────────────────

hub = AARSHub()

def cmd_ignite(args):
    hub.ignite()

def cmd_status(args):
    """Refined status display"""
    console.print(Align.center(BANNER))
    table = Table(title="Live Swarm Status", box=None)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    
    try:
        # Check vLLM 1
        q3 = hub.exec_remote("curl -s http://localhost:8000/v1/models | grep -q Qwen && echo 'ONLINE' || echo 'OFFLINE'", capture=True)
        table.add_row("🧠 Qwen3 Brain", f"[{'green' if 'ONLINE' in q3 else 'red'}]{q3}[/]")
        
        # Check vLLM 2
        qa = hub.exec_remote("curl -s http://localhost:8001/v1/models | grep -q Qwen && echo 'ONLINE' || echo 'OFFLINE'", capture=True)
        table.add_row("👂 Qwen2-Audio Hearing", f"[{'green' if 'ONLINE' in qa else 'red'}]{qa}[/]")
        
        # Check GPU
        gpu = hub.exec_remote("rocm-smi --showuse | grep -i 'GPU' | head -1", capture=True)
        table.add_row("🏎️ AMD MI300X", gpu)
        
        console.print(Panel(table, border_style="dim"))
    except:
        hub.handle_lockout()

def scp_upload(local: str, remote: str) -> int:
    cmd = f'{SCP_BASE} -r "{local}" {SERVER_USER}@{SERVER_IP}:"{remote}"'
    return subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

# ─── Additional Commands ───────────────────────────────────

def cmd_build(args):
    """Rebuild only the pipeline container."""
    console.print(Align.center(BANNER))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        METER_COLUMN,
        console=console,
    ) as progress:
        task = progress.add_task("🔨 Building Restoration Container...", total=100)
        hub.ignite_phase_4() # Build
        progress.update(task, completed=100)
    console.print("[green]  [✓] Pipeline Container Rebuilt Successfully.[/]")

def cmd_clean(args):
    """Clear reports and output on server."""
    console.print(Align.center(BANNER))
    if Confirm.ask("[red]This will delete all reports and restored files on the server. Continue?[/]"):
        hub.exec_remote("rm -f /data/reports/*.json /data/output/*")
        console.print("[green]  [✓] Server storage cleaned.[/]")

def cmd_logs(args):
    """Show container logs."""
    console.print(Align.center(BANNER))
    lines = args.lines if hasattr(args, 'lines') else 50
    
    if hasattr(args, 'pipeline') and args.pipeline:
        console.print("[bold green]── Restoration Pipeline Swarm ──[/bold green]")
        # Find active pipeline container
        container_id = hub.exec_remote("docker ps --filter 'ancestor=aars-pipeline' --format '{{.ID}}' | head -1", capture=True)
        if container_id:
            hub.exec_remote(f"docker logs --tail {lines} {container_id}", capture=False)
        else:
            console.print("[yellow]No active pipeline container found.[/yellow]")
    else:
        console.print("[bold white]── Qwen3 (brain) ──[/bold white]")
        hub.exec_remote(f"docker logs --tail {lines} vllm-qwen3 2>&1", capture=False)
        console.print("\n[bold white]── Qwen2-Audio (listener) ──[/bold white]")
        hub.exec_remote(f"docker logs --tail {lines} vllm-qwen2-audio 2>&1", capture=False)

def cmd_gpu(args):
    console.print(Align.center(BANNER))
    hub.exec_remote("rocm-smi", capture=False)

def cmd_run(args):
    """Run the audio restoration pipeline."""
    console.print(Align.center(BANNER))
    hub.log_phase(1, "Running Audio Restoration Pipeline")
    try:
        hub.exec_remote(
            "docker run --rm --device /dev/kfd --device /dev/dri --group-add video "
            "-v /data:/data "
            "-e VLLM_BASE_URL=http://localhost:8000/v1 "
            "-e VLLM_AUDIO_URL=http://localhost:8001/v1 "
            "-e HSA_OVERRIDE_GFX_VERSION=9.4.2 "
            "--network host "
            "aars-pipeline"
        )
        console.print(Panel("[bold green]PIPELINE COMPLETE[/bold green]", border_style="green"))
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")

def cmd_upload(args):
    """Upload files to server."""
    console.print(Align.center(BANNER))
    for f in args.files:
        if not os.path.exists(f):
            console.print(f"[red]Not found: {f}[/red]")
            continue
        fname = os.path.basename(f)
        console.print(f"[dim]Uploading {fname}...[/dim]")
        scp_upload(f, f"/data/input/{fname}")
        md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
        console.print(f"[green]✅ {fname} (MD5: {md5[:12]}...)[/green]")

def cmd_download(args):
    """Download completed files from server."""
    console.print(Align.center(BANNER))
    local_out = args.dest or str(PROJECT_ROOT / "output")
    os.makedirs(local_out, exist_ok=True)
    result = hub.exec_remote("ls /data/output/ 2>/dev/null", capture=True)
    if not result:
        console.print("[yellow]No completed files on server.[/yellow]")
        return
    files = [f.strip() for f in result.split("\n") if f.strip()]
    for f in files:
        console.print(f"[dim]  ▶ {f}[/dim]")
        cmd = f'{SCP_BASE} -r {SERVER_USER}@{SERVER_IP}:"/data/output/{f}" "{os.path.join(local_out, f)}"'
        subprocess.call(cmd, shell=True)
    console.print(f"[green]✅ {len(files)} file(s) saved to {local_out}[/green]")

def cmd_purge(args):
    """Kill ALL containers on the server."""
    console.print(Align.center(BANNER))
    hub.exec_remote("docker rm -f $(docker ps -aq) 2>/dev/null || true")
    console.print("[green]All containers purged.[/green]")

# ─── CLI Router ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hd.py",
        description="AARS Headquarters — Autonomous Audio Restoration Swarm"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ignite",   help="🔥 System deployment")
    sub.add_parser("build",    help="🔨 Rebuild restoration container")
    sub.add_parser("run",      help="▶  Execute restoration")
    sub.add_parser("clean",    help="🧹 Clear server reports/output")
    sub.add_parser("gpu",      help="🏎️  GPU status (rocm-smi)")
    sub.add_parser("purge",    help="🧹 Kill ALL containers")
    sub.add_parser("stop",     help="⏹️  Stop all containers gracefully")
    sub.add_parser("cost",     help="💰 Session cost estimate")
    sub.add_parser("shell",    help="🖥️  Interactive SSH")

    p_log = sub.add_parser("logs", help="📜 Container logs")
    p_log.add_argument("--lines", type=int, default=50)
    p_log.add_argument("--pipeline", action="store_true", help="Show restoration pipeline logs")

    p_up = sub.add_parser("upload", help="⬆️  Upload files to server")
    p_up.add_argument("files", nargs="+")

    p_dl = sub.add_parser("download", help="⬇️  Download completed files")
    p_dl.add_argument("--dest", default=None)

    args = parser.parse_args()

    dispatch = {
        "ignite": cmd_ignite,
        "build": cmd_build,
        "run": cmd_run,
        "clean": cmd_clean,
        "logs": cmd_logs,
        "gpu": cmd_gpu,
        "upload": cmd_upload,
        "download": cmd_download,
        "purge": cmd_purge,
    }

    if args.command in dispatch:
        try:
            dispatch[args.command](args)
        except KeyboardInterrupt:
            console.print("\n[bold red]Aborted (Ctrl+C).[/bold red]")
            sys.exit(1)
    elif args.command == "stop":
        hub.exec_remote("docker stop $(docker ps -q) 2>/dev/null || true")
        console.print("[green]All systems stopped.[/green]")
    elif args.command == "cost":
        try:
            uptime = float(hub.exec_remote("cat /proc/uptime", capture=True).split()[0])
            cost = (uptime / 3600) * RATE_PER_HOUR
            console.print(Panel(
                f"[bold yellow]Uptime: {uptime/3600:.2f} hours\nRate: ${RATE_PER_HOUR}/hr\nTotal: ${cost:.2f}[/bold yellow]",
                border_style="yellow", title="💰 Session Cost"
            ))
        except Exception:
            console.print("[red]Cannot reach server.[/red]")
    elif args.command == "shell":
        os.system(f'ssh -i "{SSH_KEY}" {SERVER_USER}@{SERVER_IP}')
    else:
        console.print(Align.center(BANNER))
        parser.print_help()

if __name__ == "__main__":
    main()
