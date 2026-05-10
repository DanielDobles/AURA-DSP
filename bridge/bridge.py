"""
AARS Bridge — Windows Edge Client
Monitors local folder for new audio files, uploads to MI300X via SFTP,
polls for completed files and downloads them back.

Usage: python bridge.py
"""

import os
import sys
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import yaml
import paramiko
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bridge.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("AARS-Bridge")

# ─── Config ────────────────────────────────────────────────
def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ─── SFTP Client ───────────────────────────────────────────
class SFTPBridge:
    """Manages SSH/SFTP connection to the MI300X Droplet."""

    def __init__(self, config: dict):
        self.host = config["server"]["host"]
        self.port = config["server"]["port"]
        self.user = config["server"]["user"]
        key_path = os.path.expanduser(config["server"]["key_path"])
        self.key_path = key_path
        self.client = None
        self.sftp = None

    def connect(self):
        """Establish SSH + SFTP connection."""
        if self.client and self.client.get_transport() and self.client.get_transport().is_active():
            return  # Already connected

        log.info(f"Connecting to {self.host}:{self.port}...")
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.host,
            port=self.port,
            username=self.user,
            key_filename=self.key_path,
            timeout=15,
        )
        self.sftp = self.client.open_sftp()
        log.info(f"Connected to {self.host}")

    def disconnect(self):
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        log.info("Disconnected.")

    def upload(self, local_path: str, remote_path: str) -> bool:
        """Upload a file via SFTP with MD5 verification."""
        try:
            self.connect()
            filename = os.path.basename(local_path)
            remote_file = f"{remote_path}/{filename}"

            # Ensure remote directory exists
            try:
                self.sftp.stat(remote_path)
            except FileNotFoundError:
                self.sftp.mkdir(remote_path)

            log.info(f"Uploading: {filename} → {remote_file}")
            self.sftp.put(local_path, remote_file)

            # MD5 verification
            local_md5 = self._local_md5(local_path)
            _, stdout, _ = self.client.exec_command(f"md5sum {remote_file}")
            remote_md5 = stdout.read().decode().split()[0]

            if local_md5 == remote_md5:
                log.info(f"✅ {filename} — hash verified: {local_md5[:12]}...")
                return True
            else:
                log.error(f"❌ {filename} — HASH MISMATCH! local={local_md5} remote={remote_md5}")
                return False

        except Exception as e:
            log.error(f"Upload failed for {local_path}: {e}")
            return False

    def download(self, remote_file: str, local_dir: str) -> bool:
        """Download a file from the server and track it locally."""
        try:
            self.connect()
            filename = os.path.basename(remote_file)
            local_path = os.path.join(local_dir, filename)

            # Avoid re-downloading if already synced
            synced_list_path = Path(__file__).parent / "synced_files.txt"
            synced_files = set()
            if synced_list_path.exists():
                synced_files = set(synced_list_path.read_text().splitlines())
            
            if filename in synced_files and os.path.exists(local_path):
                return True

            log.info(f"Downloading: {remote_file} → {local_path}")
            self.sftp.get(remote_file, local_path)

            # Verify
            local_md5 = self._local_md5(local_path)
            _, stdout, _ = self.client.exec_command(f"md5sum {remote_file}")
            remote_md5 = stdout.read().decode().split()[0]

            if local_md5 == remote_md5:
                log.info(f"✅ Downloaded {filename} — verified")
                # Track as synced instead of removing from server
                with open(synced_list_path, "a") as f:
                    f.write(f"{filename}\n")
                return True
            else:
                log.error(f"❌ Download hash mismatch for {filename}")
                return False

        except Exception as e:
            log.error(f"Download failed for {remote_file}: {e}")
            return False

    def list_remote(self, remote_dir: str) -> list:
        """List files in a remote directory."""
        try:
            self.connect()
            files = self.sftp.listdir(remote_dir)
            return [f for f in files if not f.startswith(".")]
        except Exception as e:
            log.error(f"Failed to list {remote_dir}: {e}")
            return []

    @staticmethod
    def _local_md5(filepath: str) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ─── File Watcher ──────────────────────────────────────────
class InputWatcher(FileSystemEventHandler):
    """Watches local input folder for new audio files."""

    AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aiff", ".ogg"}

    def __init__(self, bridge: SFTPBridge, remote_input: str):
        self.bridge = bridge
        self.remote_input = remote_input
        self._pending = set()  # Files being written (wait for completion)

    def on_created(self, event):
        if event.is_directory:
            return
        ext = Path(event.src_path).suffix.lower()
        if ext in self.AUDIO_EXTENSIONS:
            log.info(f"New file detected: {event.src_path}")
            # Wait for file to finish writing
            self._wait_for_stable(event.src_path)
            success = self.bridge.upload(event.src_path, self.remote_input)
            if success:
                log.info(f"🚀 {Path(event.src_path).name} sent to MI300X")

    @staticmethod
    def _wait_for_stable(filepath: str, timeout: int = 60):
        """Wait until file size stops changing (write complete)."""
        prev_size = -1
        elapsed = 0
        while elapsed < timeout:
            try:
                curr_size = os.path.getsize(filepath)
                if curr_size == prev_size and curr_size > 0:
                    return
                prev_size = curr_size
            except OSError:
                pass
            time.sleep(1)
            elapsed += 1
        log.warning(f"File {filepath} may not be fully written after {timeout}s")


# ─── Output Poller ─────────────────────────────────────────
def poll_output(bridge: SFTPBridge, remote_output: str, local_output: str, interval: int):
    """Periodically check for completed files on the server."""
    while True:
        try:
            files = bridge.list_remote(remote_output)
            if files:
                log.info(f"📥 {len(files)} completed file(s) found on server")
                for f in files:
                    remote_path = f"{remote_output}/{f}"
                    bridge.download(remote_path, local_output)
        except Exception as e:
            log.error(f"Poll error: {e}")
        time.sleep(interval)


# ─── Main ──────────────────────────────────────────────────
def main():
    config = load_config()

    # Ensure local directories exist
    local_input = config["paths"]["local_input"]
    local_output = config["paths"]["local_output"]
    os.makedirs(local_input, exist_ok=True)
    os.makedirs(local_output, exist_ok=True)

    bridge = SFTPBridge(config)
    poll_interval = config["sync"]["poll_interval_seconds"]

    # Start file watcher (upload new files)
    watcher = InputWatcher(bridge, config["paths"]["remote_input"])
    observer = Observer()
    observer.schedule(watcher, local_input, recursive=False)
    observer.start()
    log.info(f"👁️  Watching: {local_input}")

    # Start output poller (download completed files)
    import threading
    poller = threading.Thread(
        target=poll_output,
        args=(bridge, config["paths"]["remote_output"], local_output, poll_interval),
        daemon=True,
    )
    poller.start()
    log.info(f"📡 Polling server every {poll_interval}s for completed files")

    log.info("=" * 50)
    log.info("AARS Bridge is ACTIVE. Drop audio files into:")
    log.info(f"  {local_input}")
    log.info("Restored files will appear in:")
    log.info(f"  {local_output}")
    log.info("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down Bridge...")
        observer.stop()
        bridge.disconnect()
    observer.join()


if __name__ == "__main__":
    main()
