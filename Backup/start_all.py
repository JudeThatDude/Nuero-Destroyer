import subprocess
import sys
import os
import psutil
import time

MAIN_SCRIPT_PATH = "Backup/Backup.py"  # Main script
MOVEMENT_SCRIPT_PATH = "Backup/Movement.py"
#SIGHT_SCRIPT = "Backup/sight.py"

def is_script_running(script_path):
    script_name = os.path.abspath(script_path)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline:
                cmdline_str = ' '.join(cmdline)
                if script_name in cmdline_str:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_script(script_path):
    if not is_script_running(script_path):
        subprocess.Popen([sys.executable, script_path])
        print(f"Started {script_path}")
    else:
        print(f"{script_path} is already running.")

def main():
    # Start initial scripts
    start_script(MAIN_SCRIPT_PATH)
    start_script(MOVEMENT_SCRIPT_PATH)
    #start_script(SIGHT_SCRIPT)

    print("All initial scripts have been started. Exiting starter script.")

if __name__ == "__main__":
    main()