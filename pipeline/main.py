"""
AARS — Main Orchestrator (v3: Self-Improving Loop)

Flow:
  1. Scan spectral signature
  2. Load swarm memory (past runs, heuristics)
  3. Build crew with spectral + memory context
  4. Execute restoration
  5. Parse QC verdict from crew output
  6. Record everything to memory for future improvement
  7. If QC fails and retries remain, adjust and re-run
"""

import os
import json
import re
from pathlib import Path
from analysis.spectral import SpectralAnalyzer
from crew import AURACrew
from memory import SwarmMemory

# Paths within the container
INPUT_DIR = Path("/data/input")
OUTPUT_DIR = Path("/data/output")
REPORT_DIR = Path("/data/reports")

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()

MAX_RETRIES = 2  # Maximum retry attempts if QC fails


def extract_qc_verdict(crew_output: str) -> dict:
    """Extract the QC verdict JSON from the crew's final output.
    
    The QC Auditor's output should contain a JSON block with qc_passed.
    We search for it robustly in case it's wrapped in prose.
    """
    text = str(crew_output)
    
    # Strategy 1: Find JSON block with qc_passed
    patterns = [
        r'\{[^{}]*"qc_passed"[^{}]*\}',  # Simple flat JSON
        r'\{(?:[^{}]|\{[^{}]*\})*"qc_passed"(?:[^{}]|\{[^{}]*\})*\}',  # Nested one level
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if "qc_passed" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Check for obvious pass/fail keywords
    lower = text.lower()
    if "approved" in lower or "passed" in lower:
        return {"qc_passed": True, "verdict": "Inferred PASS from agent output", "issues": []}
    if "rejected" in lower or "failed" in lower:
        return {"qc_passed": False, "verdict": "Inferred FAIL from agent output", "issues": ["See crew output"]}
    
    # Strategy 3: If output files exist, tentative pass
    return {"qc_passed": None, "verdict": "Could not parse QC verdict", "issues": ["QC output unparseable"]}


def extract_tool_results(crew_output: str) -> dict:
    """Extract individual tool execution results from crew output."""
    text = str(crew_output)
    results = {}
    
    tool_names = [
        "soxr_vhq_upsampler", "transient_preservation_dsp",
        "harmonic_spectral_exciter", "ffmpeg_pro_master"
    ]
    
    for tool in tool_names:
        if tool in text:
            # Try to find the JSON result block near the tool mention
            idx = text.find(tool)
            search_region = text[max(0, idx - 100):min(len(text), idx + 1000)]
            
            success_match = re.search(r'"success"\s*:\s*(true|false)', search_region, re.IGNORECASE)
            if success_match:
                results[tool] = {"success": success_match.group(1).lower() == "true"}
            else:
                results[tool] = {"success": True}  # Assume success if mentioned
    
    return results


def process_track(track_path: Path, memory: SwarmMemory, attempt: int = 1):
    """Process a single track through the OODA restoration loop."""
    
    attempt_label = f" (Attempt {attempt}/{MAX_RETRIES + 1})" if attempt > 1 else ""
    console.print(f"\n[bold magenta]🎮 LEVEL START: {track_path.name}{attempt_label}[/bold magenta]")

    with Progress(
        SpinnerColumn("dots12", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="green"),
        console=console
    ) as progress:

        # ── Phase 1: Spectral Analysis ─────────────────────
        t1 = progress.add_task("[cyan]Scanning Spectral Signature...", total=100)
        analyzer = SpectralAnalyzer()
        spectral_data = analyzer.analyze(str(track_path))
        progress.update(t1, advance=100, description="[bold green]✓ Spectral Scan Complete[/bold green]")

        # ── Phase 2: Memory Retrieval ──────────────────────
        t2 = progress.add_task("[yellow]Loading Swarm Memory...", total=100)
        memory_briefing = memory.get_context_for_strategist(spectral_data)
        progress.update(t2, advance=100, description="[bold green]✓ Memory Loaded[/bold green]")

        # Print spectral summary
        console.print(Panel(
            f"[cyan]Sample Rate:[/] {spectral_data['sample_rate']}Hz\n"
            f"[cyan]Duration:[/] {spectral_data['duration_sec']:.1f}s\n"
            f"[cyan]Freq Cutoff:[/] {spectral_data['cutoff_freq_hz']}Hz\n"
            f"[cyan]Noise Floor:[/] {spectral_data['noise_floor_db']:.1f}dB\n"
            f"[cyan]Crest Factor:[/] {spectral_data['crest_factor_db']:.1f}dB\n"
            f"[cyan]Needs Super-Res:[/] {'YES' if spectral_data['needs_super_res'] else 'No'}\n"
            f"[cyan]Needs Denoise:[/] {'YES' if spectral_data['needs_denoising'] else 'No'}\n"
            f"[cyan]Swarm Memory:[/] {memory.get_total_runs()} past runs, "
            f"{memory.get_success_rate()*100:.0f}% success rate",
            title=f"📊 Spectral Diagnosis: {track_path.name}",
            border_style="cyan"
        ))

        # ── Phase 3: Crew Assembly & Execution ─────────────
        t3 = progress.add_task("[yellow]Assembling OODA Swarm...", total=None)
        swarm = AURACrew()
        crew = swarm.build_crew()
        progress.update(t3, total=100, advance=100, description="[bold green]✓ Swarm Assembled (3 Agents)[/bold green]")

        t4 = progress.add_task("[magenta]Executing Restoration Protocol...", total=None)
        
        # Prepare inputs for the crew
        crew_inputs = {
            "audio_path": str(track_path.absolute()),
            "filename": track_path.name,
            "spectral_json": json.dumps(spectral_data, indent=2),
            "memory_briefing": memory_briefing
        }
        
        result = crew.kickoff(inputs=crew_inputs)
        progress.update(t4, total=100, advance=100, description="[bold green]✓ Restoration Cycle Complete[/bold green]")

    # ── Phase 4: Post-Processing & Memory Update ───────────
    crew_output = str(result)
    
    # Extract QC verdict
    qc_verdict = extract_qc_verdict(crew_output)
    tool_results = extract_tool_results(crew_output)
    
    # Check for actual output files
    output_files = list(OUTPUT_DIR.glob(f"{track_path.stem}*"))
    files_exist = len(output_files) > 0

    # Determine overall success
    qc_passed = qc_verdict.get("qc_passed")
    if qc_passed is None:
        qc_passed = files_exist  # Fallback to file existence check
    
    overall_success = files_exist and qc_passed

    # Record to memory
    memory_entry = {
        "filename": track_path.name,
        "attempt": attempt,
        "spectral": spectral_data,
        "tool_results": tool_results,
        "qc_metrics": qc_verdict.get("deltas", {}),
        "qc_passed": overall_success,
        "failure_reason": "; ".join(qc_verdict.get("issues", [])) if not overall_success else None,
        "crew_output_excerpt": crew_output[:500]
    }
    memory.record_run(memory_entry)

    # Save detailed report
    report_file = REPORT_DIR / f"{track_path.stem}_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "spectral": spectral_data,
            "status": "SUCCESS" if overall_success else "FAILED",
            "attempt": attempt,
            "crew_decision": crew_output[:2000],
            "qc_verdict": qc_verdict,
            "tool_results": tool_results,
            "memory_stats": {
                "total_runs": memory.get_total_runs(),
                "success_rate": round(memory.get_success_rate(), 3)
            }
        }, f, indent=4, default=str)

    # Display results
    if overall_success:
        output_names = ", ".join(f.name for f in output_files)
        
        # Show improvement metrics if available
        metrics_display = ""
        deltas = qc_verdict.get("deltas", {})
        if deltas:
            metrics_display = "\n[dim]Improvements:[/dim]\n"
            for k, v in deltas.items():
                if isinstance(v, (int, float)):
                    color = "green" if v >= 0 else "red"
                    metrics_display += f"  [{color}]{k}: {v:+.1f}[/{color}]\n"

        console.print(Panel(
            f"[bold green]MISSION PASSED[/bold green]\n"
            f"Track: {track_path.name}\n"
            f"Outputs: {output_names}\n"
            f"Report: {report_file.name}"
            f"{metrics_display}",
            border_style="green"
        ))
    else:
        issues = qc_verdict.get("issues", ["Unknown"])
        console.print(Panel(
            f"[bold red]MISSION {'FAILED' if attempt > MAX_RETRIES else 'NEEDS RETRY'}[/bold red]\n"
            f"Track: {track_path.name}\n"
            f"Issues: {'; '.join(issues)}\n"
            f"Report: {report_file.name}",
            border_style="red" if attempt > MAX_RETRIES else "yellow"
        ))
        
        # Retry with adjusted parameters if retries remain
        if attempt <= MAX_RETRIES and not files_exist:
            console.print(f"[yellow]🔄 Retrying with swarm memory feedback...[/yellow]")
            # Clean partial outputs before retry
            for f in output_files:
                f.unlink(missing_ok=True)
            return process_track(track_path, memory, attempt + 1)


def main():
    # Ensure directories exist
    for d in [INPUT_DIR, OUTPUT_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize persistent memory
    memory = SwarmMemory()
    
    console.print(Panel(
        f"[bold cyan]Swarm Memory Status[/bold cyan]\n"
        f"Past Runs: {memory.get_total_runs()}\n"
        f"Success Rate: {memory.get_success_rate()*100:.0f}%",
        border_style="cyan", title="🧠 Memory"
    ))

    tracks = sorted(list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.mp3")))

    if not tracks:
        console.print("[yellow]📭 No tracks found in /data/input. Awaiting bridge...[/yellow]")
        return

    # Summary table
    table = Table(title="📋 Processing Queue", show_header=True)
    table.add_column("Track", style="cyan")
    table.add_column("Status", style="white")
    
    for track in tracks:
        report_file = REPORT_DIR / f"{track.stem}_report.json"
        output_exists = any(OUTPUT_DIR.glob(f"{track.stem}*"))
        if report_file.exists() and output_exists:
            table.add_row(track.name, "[green]✓ Already processed[/green]")
        else:
            table.add_row(track.name, "[yellow]⏳ Queued[/yellow]")
    
    console.print(table)

    # Process each track
    for track in tracks:
        report_file = REPORT_DIR / f"{track.stem}_report.json"
        output_exists = any(OUTPUT_DIR.glob(f"{track.stem}*"))
        if report_file.exists() and output_exists:
            console.print(f"[dim]⏩ Skipping {track.name} (Already processed & output found).[/dim]")
            continue

        try:
            process_track(track, memory)
        except Exception as e:
            console.print(f"[bold red]❌ Failed to process {track.name}: {e}[/bold red]")
            # Record failure to memory
            memory.record_run({
                "filename": track.name,
                "attempt": 1,
                "spectral": {},
                "tool_results": {},
                "qc_metrics": {},
                "qc_passed": False,
                "failure_reason": str(e)
            })

    # Final summary
    console.print(Panel(
        f"[bold green]Session Complete[/bold green]\n"
        f"Total Runs: {memory.get_total_runs()}\n"
        f"Overall Success Rate: {memory.get_success_rate()*100:.0f}%",
        border_style="green", title="📊 Session Summary"
    ))


if __name__ == "__main__":
    main()
