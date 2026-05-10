"""
╔══════════════════════════════════════════════════════════════╗
║  AARS HQ — The Hub (v3 Autonomous Orchestrator)              ║
║  One-command deployment and restoration control.              ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import hashlib
import math
import os
import subprocess
import sys
import time
import threading
import queue
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

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
    from rich import box
    HAS_RICH = True
except ImportError:
    print("Error: 'rich' library not found. Run: pip install rich")
    sys.exit(1)

console = Console()

# ─── Configuration ─────────────────────────────────────────
SERVER_IP = os.getenv("DROPLET_IP", "0.0.0.0")
SERVER_USER = "root"
SSH_KEY = os.path.expanduser(os.getenv("SSH_KEY_PATH", "~/.ssh/id_gpu_droplet"))

SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=60 -o ConnectionAttempts=5"
SSH_BASE = f'ssh {SSH_OPTS} -i "{SSH_KEY}" {SERVER_USER}@{SERVER_IP}'
SCP_BASE = f'scp {SSH_OPTS} -i "{SSH_KEY}"'

PROJECT_ROOT = Path(__file__).parent
INFRA_DIR = PROJECT_ROOT / "infra"
BRIDGE_DIR = PROJECT_ROOT / "bridge"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

RATE_PER_HOUR = 1.99

# Audio file extensions recognized by the scanner
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".ogg", ".aac", ".wma", ".m4a"}

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

    def exec_remote(self, cmd: str, capture: bool = False, retries: int = 2, status_msg: str = None, quiet: bool = False):
        full_cmd = f'{SSH_BASE} "{cmd}"'
        display_msg = status_msg or f"Executing: [dim]{cmd[:50]}...[/dim]"
        
        for i in range(retries + 1):
            try:
                if capture:
                    if quiet:
                        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, check=True)
                        return result.stdout.strip()
                    else:
                        with console.status(f"[cyan]{display_msg}[/cyan]", spinner="dots"):
                            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, check=True)
                            return result.stdout.strip()
                else:
                    # Stream output for better UX
                    return self.exec_remote_stream(full_cmd, display_msg)
            except subprocess.CalledProcessError as e:
                if i < retries and e.returncode == 255:
                    console.print(f"[dim]  (SSH retry {i+1}/{retries}...)[/]")
                    time.sleep(2)
                    continue
                err_msg = e.stderr if e.stderr else str(e)
                if "Connection timed out" in err_msg or "Permission denied" in err_msg:
                    self.handle_lockout()
                raise e

    def exec_remote_stream(self, full_cmd: str, status_msg: str = "Processing...") -> bool:
        """Runs a remote command and streams output to console with a live view."""
        from rich.live import Live
        from rich.table import Table
        from rich.spinner import Spinner
        
        lines_to_show = 8
        output_buffer = []
        
        def get_renderable():
            table = Table.grid(expand=True)
            # Use a real moving spinner
            table.add_row(f"[bold cyan]{Spinner('dots', style='cyan').render(time.time())}[/] [white]{status_msg}[/white]")
            for line in output_buffer[-lines_to_show:]:
                table.add_row(f"  [dim]↳[/] [grey50]{line}[/]")
            return Panel(table, border_style="cyan", padding=(0, 1))

        with Live(get_renderable(), refresh_per_second=12) as live:
            process = subprocess.Popen(
                full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1
            )
            
            # Non-blocking read using a thread and queue
            q = queue.Queue()
            def reader():
                for char in iter(lambda: process.stdout.read(1), ''):
                    q.put(char)
                process.stdout.close()
            
            t = threading.Thread(target=reader, daemon=True)
            t.start()
            
            partial_line = ""
            while True:
                # Try to get characters from the queue without blocking for too long
                try:
                    while True:
                        char = q.get_nowait()
                        if char == '\n' or char == '\r':
                            if partial_line.strip():
                                if char == '\r' and output_buffer and len(partial_line) > 10:
                                    output_buffer[-1] = partial_line.strip()
                                else:
                                    output_buffer.append(partial_line.strip())
                            partial_line = ""
                        else:
                            partial_line += char
                except queue.Empty:
                    pass
                
                # Update spinner/live view
                live.update(get_renderable())
                
                if not t.is_alive() and q.empty():
                    break
                
                time.sleep(0.05)
            
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
            progress.update(task1, description="[cyan]Verifying SSH Handshake...")
            try:
                hub.exec_remote("echo 1", capture=True, status_msg="Checking SSH connection")
            except:
                self.handle_lockout()
            progress.advance(task1)
            
            # --- PHASE 2: Infrastructure ---
            task2 = progress.add_task("[magenta]PHASE 2: Hardening MI300X...", total=3)
            
            # 2.1 Upload scripts (Single connection)
            progress.update(task2, description="[magenta]Preparing remote environment...")
            self.exec_remote("mkdir -p /tmp/aars", capture=True, status_msg="Creating /tmp/aars")
            if not scp_upload(str(INFRA_DIR), "/tmp/aars/"):
                console.print("[bold red]Critical Error: Failed to upload infrastructure scripts.[/bold red]")
                sys.exit(1)
            progress.advance(task2)
            
            # 2.2 Run Setup
            progress.update(task2, description="[magenta]Installing OS dependencies (Docker, ROCm)...")
            try:
                self.exec_remote("chmod +x /tmp/aars/infra/*.sh && bash /tmp/aars/infra/setup_server.sh", 
                                 status_msg="Running Server Setup (this may take a while)")
            except subprocess.CalledProcessError as e:
                if e.returncode == 255:
                    console.print("[yellow]Notice: Connection dropped during SSH reload. This is expected. Re-verifying...[/yellow]")
                    time.sleep(2)
                    self.exec_remote("echo 1", capture=True, status_msg="Re-verifying connection") # Verify re-connection
                else:
                    raise e
            progress.advance(task2)
            
            # --- PHASE 3: vLLM Swarm ---
            task3 = progress.add_task("[yellow]PHASE 3: Deploying Qwen Swarm...", total=3)
            
            # Check if models are ALREADY running to save time
            progress.update(task3, description="[yellow]Probing for existing Swarm models...")
            q3_online = False
            try:
                res = self.exec_remote("curl -s http://localhost:8000/v1/models | grep -q Qwen && echo 1 || echo 0", 
                                       capture=True, status_msg="Checking Qwen3 (Port 8000)")
                q3_online = "1" in res
            except: pass
            
            qa_online = False
            try:
                res = self.exec_remote("curl -s http://localhost:8001/v1/models | grep -q Qwen && echo 1 || echo 0", 
                                       capture=True, status_msg="Checking Qwen2-Audio (Port 8001)")
                qa_online = "1" in res
            except: pass
            
            if q3_online and qa_online:
                console.print("[green]  [✓] vLLM models already online. Skipping redeploy.[/]")
                progress.advance(task3, 2)
            else:
                # 3.0 Purge ALL old containers (AURYGA leftovers + previous runs)
                progress.update(task3, description="[yellow]Cleaning up legacy containers...")
                self.exec_remote("docker rm -f $(docker ps -aq) 2>/dev/null || true", 
                                 capture=True, status_msg="Purging Docker containers")
                progress.advance(task3)
                
                # 3.1 Deploy fresh
                progress.update(task3, description="[yellow]Waking up Brain (Qwen3) & Hearing (Qwen2-Audio)...")
                try:
                    self.exec_remote("bash /tmp/aars/infra/deploy_vllm.sh", 
                                     status_msg="Deploying vLLM containers")
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
        self.exec_remote("mkdir -p /opt/aars/pipeline", capture=True, status_msg="Preparing pipeline workspace")
        if not scp_upload(str(PIPELINE_DIR), "/opt/aars/"):
            console.print("[bold red]Critical Error: Failed to upload pipeline source.[/bold red]")
            sys.exit(1)
        self.exec_remote("docker build -t aars-pipeline /opt/aars/pipeline/", 
                         status_msg="Building Pipeline Container (this takes a moment)")

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
    
    # Also show the telemetry dashboard
    _cmd_telemetry(args)

def scp_upload(local: str, remote: str) -> bool:
    """Uploads a file or directory via SCP and returns True if successful."""
    cmd = f'{SCP_BASE} -r "{local}" {SERVER_USER}@{SERVER_IP}:"{remote}"'
    with console.status(f"[magenta]Transferring {Path(local).name}...[/magenta]", spinner="bouncingBar"):
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode == 0

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

# ─── Adaptive Onboarding System ────────────────────────────

def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    return f"{s} {units[i]}"


def scan_input_folder(input_path: str) -> list:
    """Scan a local folder recursively for audio files.
    
    Returns a list of dicts with: path, name, format, size_bytes, valid, warning
    The system adapts to whatever the user has — 1 song or 1000.
    """
    tracks = []
    root = Path(input_path)
    
    if not root.exists():
        console.print(f"[bold red]❌ Ruta no encontrada: {input_path}[/bold red]")
        return tracks
    
    # If it's a single file, just process that one
    if root.is_file():
        ext = root.suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            size = root.stat().st_size
            tracks.append({
                "path": str(root),
                "name": root.name,
                "format": ext.upper().lstrip("."),
                "size_bytes": size,
                "valid": size > 0,
                "warning": "Archivo vacío (0 bytes)" if size == 0 else None
            })
        return tracks
    
    # Recursive scan of directory
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            try:
                size = f.stat().st_size
                warning = None
                valid = True
                
                if size == 0:
                    warning = "Archivo vacío (0 bytes)"
                    valid = False
                elif size < 1024:  # Less than 1KB is suspicious
                    warning = "Archivo sospechosamente pequeño"
                    valid = False
                
                tracks.append({
                    "path": str(f),
                    "name": f.name,
                    "format": f.suffix.upper().lstrip("."),
                    "size_bytes": size,
                    "valid": valid,
                    "warning": warning
                })
            except OSError as e:
                tracks.append({
                    "path": str(f),
                    "name": f.name,
                    "format": f.suffix.upper().lstrip("."),
                    "size_bytes": 0,
                    "valid": False,
                    "warning": f"No se pudo leer: {e}"
                })
    
    return tracks


def display_track_summary(tracks: list, auto_yes: bool = False) -> bool:
    """Display an adaptive summary based on how many tracks were found.
    
    - 0 tracks: Error message
    - 1-10: Full table with every track
    - 11-50: Truncated table (first 5 + last 5)
    - 51+: Statistical summary with cost estimate
    
    Returns True if user confirms processing.
    """
    count = len(tracks)
    
    if count == 0:
        console.print(Panel(
            "[bold red]⚠ No se encontraron archivos de audio[/bold red]\n\n"
            "Formatos soportados: WAV, MP3, FLAC, AIFF, OGG, AAC, WMA, M4A\n"
            "Verifica que la ruta contenga archivos de audio válidos.",
            border_style="red", title="Sin Canciones"
        ))
        return False
    
    total_size = sum(t["size_bytes"] for t in tracks)
    valid_count = sum(1 for t in tracks if t["valid"])
    warning_count = count - valid_count
    format_counts = Counter(t["format"] for t in tracks)
    
    # ── Caso 1-10: Tabla completa ──────────────────────────
    if count <= 10:
        table = Table(
            title=f"🎵 {count} Canción{'es' if count > 1 else ''} Detectada{'s' if count > 1 else ''}",
            box=box.ROUNDED,
            show_lines=True
        )
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Nombre", style="cyan", max_width=45)
        table.add_column("Formato", style="magenta", justify="center")
        table.add_column("Tamaño", style="white", justify="right")
        table.add_column("Estado", justify="center")
        
        for i, t in enumerate(tracks, 1):
            status = "[green]✓ Válido[/green]" if t["valid"] else f"[yellow]⚠ {t['warning']}[/yellow]"
            table.add_row(
                str(i),
                t["name"],
                t["format"],
                _format_size(t["size_bytes"]),
                status
            )
        
        console.print(table)
    
    # ── Caso 11-50: Tabla truncada ─────────────────────────
    elif count <= 50:
        table = Table(
            title=f"🎵 {count} Canciones Detectadas",
            box=box.ROUNDED,
            show_lines=True
        )
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Nombre", style="cyan", max_width=45)
        table.add_column("Formato", style="magenta", justify="center")
        table.add_column("Tamaño", style="white", justify="right")
        table.add_column("Estado", justify="center")
        
        # First 5
        for i, t in enumerate(tracks[:5], 1):
            status = "[green]✓[/green]" if t["valid"] else f"[yellow]⚠[/yellow]"
            table.add_row(str(i), t["name"], t["format"], _format_size(t["size_bytes"]), status)
        
        # Ellipsis row
        hidden = count - 10
        table.add_row("...", f"[dim]... y {hidden} canciones más ...[/dim]", "", "", "")
        
        # Last 5
        for i, t in enumerate(tracks[-5:], count - 4):
            status = "[green]✓[/green]" if t["valid"] else f"[yellow]⚠[/yellow]"
            table.add_row(str(i), t["name"], t["format"], _format_size(t["size_bytes"]), status)
        
        console.print(table)
    
    # ── Caso 51+: Resumen estadístico ──────────────────────
    else:
        # Estimate processing time: ~1 min per track (conservative)
        est_hours = count / 60.0
        est_cost = est_hours * RATE_PER_HOUR
        
        format_str = ", ".join(f"{fmt} ({cnt})" for fmt, cnt in format_counts.most_common())
        
        summary_lines = [
            f"[bold cyan]📊 Resumen del Lote[/bold cyan]",
            f"{'─' * 35}",
            f"[white]Canciones:[/white]  [bold]{count}[/bold]",
            f"[white]Formatos:[/white]   {format_str}",
            f"[white]Tamaño total:[/white] [bold]{_format_size(total_size)}[/bold]",
            f"",
            f"[white]⏱ Tiempo estimado:[/white] [bold yellow]~{est_hours:.1f} horas[/bold yellow]",
            f"[white]💰 Costo estimado:[/white] [bold yellow]${est_cost:.2f} USD[/bold yellow]",
        ]
        
        if warning_count > 0:
            summary_lines.append(f"")
            summary_lines.append(f"[yellow]⚠ {warning_count} archivo{'s' if warning_count > 1 else ''} con advertencias[/yellow]")
        
        console.print(Panel(
            "\n".join(summary_lines),
            title=f"🎵 Lote Detectado: {count} Canciones",
            border_style="cyan"
        ))
    
    # ── Footer común ───────────────────────────────────────
    if count <= 50:
        console.print(f"  [dim]Total: {count} canción{'es' if count > 1 else ''} | {_format_size(total_size)}[/dim]")
        if warning_count > 0:
            console.print(f"  [yellow]⚠ {warning_count} archivo{'s' if warning_count > 1 else ''} con advertencias (se omitirán)[/yellow]")
    
    # ── Confirmación ───────────────────────────────────────
    if auto_yes:
        console.print(f"[dim]  (--yes activo, procesando automáticamente)[/dim]")
        return True
    
    prompt_text = f"¿Procesar {'esta canción' if count == 1 else f'estas {valid_count} canciones'}?"
    return Confirm.ask(prompt_text)


def upload_with_progress(tracks: list) -> dict:
    """Upload valid audio files one by one with a progress bar.
    
    Returns stats dict: {success: int, failed: int, skipped: int, total: int}
    """
    valid_tracks = [t for t in tracks if t["valid"]]
    stats = {"success": 0, "failed": 0, "skipped": len(tracks) - len(valid_tracks), "total": len(tracks)}
    
    if not valid_tracks:
        console.print("[bold red]No hay archivos válidos para subir.[/bold red]")
        return stats
    
    # Ensure remote input dir exists
    hub.exec_remote("rm -rf /data/input_dynamic && mkdir -p /data/input_dynamic", capture=True,
                    status_msg="Preparando directorio remoto")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        METER_COLUMN,
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Subiendo {len(valid_tracks)} canciones...",
            total=len(valid_tracks)
        )
        
        for i, t in enumerate(valid_tracks, 1):
            progress.update(task, description=f"[cyan]Subiendo {i}/{len(valid_tracks)}: {t['name']}")
            
            if scp_upload(t["path"], "/data/input_dynamic/"):
                stats["success"] += 1
            else:
                stats["failed"] += 1
                console.print(f"  [red]❌ Falló: {t['name']}[/red]")
            
            progress.advance(task)
    
    # Upload summary
    summary_table = Table(title="📤 Resumen de Upload", box=box.ROUNDED)
    summary_table.add_column("Estado", style="white")
    summary_table.add_column("Cantidad", justify="right")
    
    summary_table.add_row("[green]✅ Subidos exitosamente[/green]", f"[bold green]{stats['success']}[/bold green]")
    if stats["failed"] > 0:
        summary_table.add_row("[red]❌ Fallidos[/red]", f"[bold red]{stats['failed']}[/bold red]")
    if stats["skipped"] > 0:
        summary_table.add_row("[yellow]⚠ Omitidos (inválidos)[/yellow]", f"[yellow]{stats['skipped']}[/yellow]")
    
    console.print(summary_table)
    
    return stats


# ─── Run Command (Adaptive Onboarding) ────────────────────

def cmd_run(args):
    """Run the audio restoration pipeline with adaptive onboarding.
    
    The CLI adapts to the user's input:
    - Scans the input folder for audio files
    - Shows a summary adapted to the volume (1, 10, or 1000 songs)
    - Uploads with per-file progress
    - Launches the restoration swarm
    """
    console.print(Align.center(BANNER))
    
    input_path = getattr(args, "input", "/data/input")
    auto_yes = getattr(args, "yes", False)
    track_count_label = None  # Will be set if adaptive onboarding runs
    
    # Check if input_path is a Windows path. If so, run the adaptive onboarding.
    if ":" in input_path or "\\" in input_path:
        while True:
            # Phase 1: Scan with gaming-style loading UI
            with console.status(f"[bold cyan]👾 Analizando radar local:[/bold cyan] Escaneando {input_path} en busca de pistas de audio...", spinner="bouncingBar"):
                tracks = scan_input_folder(input_path)
            
            # Phase 2: Display adaptive summary & confirm
            if display_track_summary(tracks, auto_yes=auto_yes):
                break # User confirmed, proceed to upload
            
            # User declined or no tracks found. Ask for a new path.
            console.print("\n[yellow]¿Deseas intentar con otra carpeta?[/yellow]")
            try:
                new_path = Prompt.ask("[cyan]Introduce la nueva ruta (o presiona Ctrl+C para abortar)[/cyan]")
                input_path = new_path.strip('"\'')
                # If they provide a non-Windows path, break to let it run remotely
                if not (":" in input_path or "\\" in input_path):
                    break
            except KeyboardInterrupt:
                console.print("\n[dim]Operación cancelada por el usuario.[/dim]")
                return
                
        # If input_path is still a Windows path, it means they confirmed the tracks
        if ":" in input_path or "\\" in input_path:
            # Phase 3: Upload with progress
            console.print("\n[bold magenta]📡 Transfiriendo al MI300X...[/bold magenta]")
            upload_stats = upload_with_progress(tracks)
            
            if upload_stats["success"] == 0:
                console.print("[bold red]Error: No se pudo subir ningún archivo.[/bold red]")
                return
            
            track_count_label = f"{upload_stats['success']} tracks"
            # Set the remote path for the pipeline
            input_path = "/data/input_dynamic"
    
    do_purge = " --purge" if getattr(args, "purge", False) else ""
    do_overwrite = " --overwrite" if getattr(args, "overwrite", False) else ""
    status_label = track_count_label or f"Input: {input_path}"
    
    try:
        hub.exec_remote(
            f"docker run --rm --device /dev/kfd --device /dev/dri --group-add video "
            f"-v {input_path}:/data/input "
            f"-v /data/output:/data/output "
            f"-v /data/reports:/data/reports "
            f"-v /data/intermediate:/data/intermediate "
            f"-e VLLM_BASE_URL=http://127.0.0.1:8000/v1 "
            f"-e VLLM_AUDIO_URL=http://127.0.0.1:8001/v1 "
            f"-e HSA_OVERRIDE_GFX_VERSION=9.4.2 "
            f"--network host "
            f"aars-pipeline python3 main.py --input /data/input{do_purge}{do_overwrite}",
            status_msg=f"Starting Audio Restoration Swarm ({status_label})"
        )
        console.print(Panel("[bold green]PIPELINE COMPLETE[/bold green]", border_style="green"))
        
        # Auto-fetch results
        console.print("\n[bold cyan]🔄 Auto-fetching results...[/bold cyan]")
        if not hasattr(args, 'dest'):
            args.dest = None
        cmd_download(args)
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")

def cmd_process(args):
    """Shortcut for dynamic processing with purge and overwrite (Recommended)."""
    args.purge = True
    args.overwrite = True
    cmd_run(args)

def cmd_upload(args):
    """Upload files to server with verification."""
    console.print(Align.center(BANNER))
    success_count = 0
    fail_count = 0
    
    for f in args.files:
        if not os.path.exists(f):
            console.print(f"[red]Not found: {f}[/red]")
            fail_count += 1
            continue
            
        fname = os.path.basename(f)
        console.print(f"[dim]Uploading {fname}...[/dim]")
        
        if scp_upload(f, f"/data/input/{fname}"):
            md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
            console.print(f"[green]✅ {fname} uploaded (MD5: {md5[:12]}...)[/green]")
            success_count += 1
        else:
            console.print(f"[red]❌ Failed to upload {fname}[/red]")
            fail_count += 1
            
    if fail_count == 0:
        console.print(f"\n[bold green]✅ All {success_count} files uploaded successfully.[/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠ Upload finished with errors: {success_count} success, {fail_count} failed.[/bold yellow]")

def _cmd_telemetry(args):
    """Show a dashboard of processed vs local files."""
    console.print(Align.center(BANNER))
    hub.log_phase(1, "AARS Global Status")
    
    # Remote files with loading UI
    with console.status("[bold magenta]📡 Sincronizando telemetría del enjambre...[/bold magenta]", spinner="bouncingBar"):
        remote_input = hub.exec_remote("ls /data/input/ 2>/dev/null", capture=True)
        remote_output = hub.exec_remote("ls /data/output/ 2>/dev/null", capture=True)
        remote_reports = hub.exec_remote("ls /data/reports/ 2>/dev/null", capture=True)
    
    # Local files
    local_output = list((PROJECT_ROOT / "output").glob("*"))
    
    table = Table(title="Mission Telemetry", box=box.ROUNDED)
    table.add_column("Track", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Remote Size", justify="right")
    table.add_column("Local Sync", justify="center")
    
    inputs = [f.strip() for f in remote_input.split("\n") if f.strip()]
    outputs = [f.strip() for f in remote_output.split("\n") if f.strip()]
    
    for track in sorted(inputs):
        stem = Path(track).stem
        restored_name = f"{stem}_restored.wav"
        
        status = "[yellow]PENDING[/]"
        size = "-"
        local = "[red]✗[/]"
        
        if restored_name in outputs:
            status = "[green]RESTORED[/]"
            size_raw = hub.exec_remote(f"du -sh /data/output/{restored_name} | cut -f1", capture=True).strip()
            size = size_raw if size_raw else "?"
            
            if any(f.name == restored_name for f in local_output):
                local = "[green]✓[/]"
        
        table.add_row(track, status, size, local)
        
    console.print(table)
    console.print(f"\n[dim]Total Local Files: {len(local_output)} | Reports: {len(remote_reports.split())}[/dim]")

def cmd_download(args):
    """Download completed files from server with error checking and race condition handling."""
    console.print(Align.center(BANNER))
    local_out = args.dest or str(PROJECT_ROOT / "output")
    os.makedirs(local_out, exist_ok=True)
    
    hub.log_phase(1, f"Syncing results to {local_out}")
    
    try:
        with console.status("[bold cyan]👾 Explorando servidor base:[/bold cyan] Buscando transmisiones restauradas...", spinner="bouncingBar"):
            result = hub.exec_remote("ls --color=never /data/output/ 2>/dev/null", capture=True)
    except Exception as e:
        console.print(f"[red]Error listing remote files: {e}[/red]")
        return

    if not result:
        console.print("[yellow]No completed files found on server.[/yellow]")
        return
        
    files = [f.strip() for f in result.split("\n") if f.strip()]
    
    stats = {"success": 0, "failed": 0, "skipped": 0, "missing": 0}
    failed_details = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        METER_COLUMN,
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Downloading tracks...", total=len(files))
        for f in files:
            local_path = os.path.join(local_out, f)
            progress.update(task, description=f"Fetching {f}...")
            
            # Check if file exists locally to avoid re-downloading
            if os.path.exists(local_path):
                 stats["skipped"] += 1
                 progress.advance(task)
                 continue
                 
            cmd = f'{SCP_BASE} -q -r {SERVER_USER}@{SERVER_IP}:"/data/output/{f}" "{local_path}"'
            # Use subprocess.run to capture result
            scp_res = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
            
            if scp_res.returncode == 0:
                stats["success"] += 1
            else:
                # Handle disappearance (Race condition with bridge)
                try:
                    # Check if it still exists on remote
                    hub.exec_remote(f"ls --color=never /data/output/{f}", capture=True, quiet=True)
                    # If it exists but SCP failed, it's a real download error
                    stats["failed"] += 1
                    failed_details.append(f"{f} (SCP error)")
                except subprocess.CalledProcessError:
                    # File is gone from server!
                    stats["missing"] += 1
                    failed_details.append(f"{f} (Disappeared - likely moved by bridge)")
            
            progress.advance(task)
            
    # Final Summary Report
    summary = Table(title="Sync Results", box=box.ROUNDED)
    summary.add_column("Category", style="cyan")
    summary.add_column("Count", justify="right")
    
    summary.add_row("Successfully Downloaded", f"[green]{stats['success']}[/]")
    if stats["skipped"]: summary.add_row("Already Local (Skipped)", f"[blue]{stats['skipped']}[/]")
    if stats["missing"]: summary.add_row("Missing from Server", f"[yellow]{stats['missing']}[/]")
    if stats["failed"]:  summary.add_row("Download Failed", f"[bold red]{stats['failed']}[/]")
    
    console.print(summary)
    
    if stats["failed"] == 0:
        if stats["missing"] > 0:
            console.print(f"\n[bold yellow]⚠ Sync Complete (with caveats).[/bold yellow] Some files were missing from the server, probably already handled by the AARS Bridge.")
        else:
            console.print(f"\n[bold green]✅ Sync Complete![/bold green] All files synced successfully.")
    else:
        console.print(f"\n[bold red]❌ Sync Failed for {stats['failed']} files.[/bold red]")
        for detail in failed_details:
            console.print(f"  [red]↳[/] {detail}")

def cmd_purge(args):
    """Kill ALL containers on the server."""
    console.print(Align.center(BANNER))
    hub.exec_remote("docker rm -f $(docker ps -aq) 2>/dev/null || true")
    console.print("[green]All containers purged.[/green]")

# ─── CLI Router ────────────────────────────────────────────

def main():
    # Parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm all prompts")

    parser = argparse.ArgumentParser(
        prog="hd.py",
        description="AARS Headquarters — Autonomous Audio Restoration Swarm"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ignite",   parents=[parent_parser], help="🔥 System deployment")
    sub.add_parser("build",    parents=[parent_parser], help="🔨 Rebuild restoration container")
    
    p_run = sub.add_parser("run", parents=[parent_parser], help="▶  Execute restoration")
    p_run.add_argument("--input", default="C:\\Suno_Restoration\\Input", help="Path to process (e.g. C:\\Suno_Restoration\\Input or /mnt/scratch/my_songs)")
    p_run.add_argument("--purge", action="store_true", help="Purge temporary files for a fresh start")
    p_run.add_argument("--overwrite", action="store_true", help="Overwrite existing output files and reports")

    p_proc = sub.add_parser("process", parents=[parent_parser], help="🔥 Fresh dynamic processing (Recommended)")
    p_proc.add_argument("input", nargs="?", default="C:\\Suno_Restoration\\Input", help="Path to process")
    
    sub.add_parser("status",   parents=[parent_parser], help="📊 Mission telemetry & local sync status")
    sub.add_parser("clean",    parents=[parent_parser], help="🧹 Clear server reports/output")
    sub.add_parser("gpu",      parents=[parent_parser], help="🏎️  GPU status (rocm-smi)")
    sub.add_parser("purge",    parents=[parent_parser], help="🧹 Kill ALL containers")
    sub.add_parser("stop",     parents=[parent_parser], help="⏹️  Stop all containers gracefully")
    sub.add_parser("cost",     parents=[parent_parser], help="💰 Session cost estimate")
    sub.add_parser("shell",    parents=[parent_parser], help="🖥️  Interactive SSH")

    p_log = sub.add_parser("logs", parents=[parent_parser], help="📜 Container logs")
    p_log.add_argument("--lines", type=int, default=50)
    p_log.add_argument("--pipeline", action="store_true", help="Show restoration pipeline logs")

    p_up = sub.add_parser("upload", parents=[parent_parser], help="⬆️  Upload files to server")
    p_up.add_argument("files", nargs="+")

    p_dl = sub.add_parser("fetch", parents=[parent_parser], help="⬇️  Download restored files (alias: download)")
    p_dl.add_argument("--dest", default=None)
    
    # Also keep 'download' as an alias if needed, but fetch is better
    sub.add_parser("download", parents=[parent_parser], help="⬇️  Download restored files")

    args = parser.parse_args()

    dispatch = {
        "ignite": cmd_ignite,
        "build": cmd_build,
        "run": cmd_run,
        "process": cmd_process,
        "status": cmd_status,
        "clean": cmd_clean,
        "logs": cmd_logs,
        "gpu": cmd_gpu,
        "upload": cmd_upload,
        "download": cmd_download,
        "fetch": cmd_download,
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
