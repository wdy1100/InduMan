import os
import numpy as np
import tkinter as tk
import threading
import queue
import time

from pathlib import Path

def get_induman_path() -> Path:
    """
    Get the path to the 'InduMan' project root directory.
    
    This function searches upward from the current file's location
    until it finds a directory named 'InduMan'.
    
    Returns:
        Path: Absolute path to the 'InduMan' directory.
        
    Raises:
        FileNotFoundError: If 'InduMan' directory is not found in the hierarchy.
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if parent.name == "InduMan":
            return parent
    raise FileNotFoundError("Could not locate 'InduMan' directory in the path hierarchy.")

def get_assets_path():
    """
    get our assets folder path
    """
    # get current directory where this file is located
    current_dir = get_induman_path()
    # join with assets folder name
    assets_dir = os.path.join(current_dir, "assets")
    
    return assets_dir

def get_task_cfg_path():
    """
    get task config file path
    """
    # get current directory where this file is located
    current_dir = get_induman_path()
    # join with assets folder name
    assets_dir = os.path.join(current_dir, "task_config")
    
    return assets_dir    

def get_default_data_path():
    """
    get default data folder path
    """
    # get current directory where this file is located
    current_dir = get_induman_path()
    # join with assets folder name
    data_dir = os.path.join(current_dir, "record_data")
    
    return data_dir

class ForceMonitor:
    def __init__(self, threshold=3.0, min_interval=1.0):
        """
        threshold: force threshold (absolute value)
        min_interval: minimum reminder interval (in seconds) for the same object
        """
        self.threshold = threshold
        self.min_interval = min_interval
        self._last_popup_time = {}
        self._gui_queue = queue.Queue()
        self._current_popup = None

        # start GUI thread
        self._gui_thread = threading.Thread(target=self._gui_mainloop, daemon=True)
        self._gui_thread.start()

    # ---------------- Force Detection ----------------
    def check_observation(self, observation: dict):
        """
        Check all keys in observation that end with 'contact_forces'.
        """
        for key, forces in observation.items():
            if not key.endswith("contact_forces"):
                continue
            try:
                forces = np.array(forces, dtype=float).reshape(-1)
            except Exception:
                continue

            # check if any force exceeds the threshold
            if np.any(np.abs(forces) > self.threshold):
                now = time.time()
                if now - self._last_popup_time.get(key, 0) >= self.min_interval:
                    self._last_popup_time[key] = now
                    self._alert(key, forces)

    def _alert(self, obj_name, forces):
        """Print warning and post to GUI popup"""
        RED = "\033[91m"
        BOLD = "\033[1m"
        END = "\033[0m"
        print(f"{BOLD}{RED}⚠️ Large contact force → {obj_name} : {forces}{END}")

        self._post_to_gui(self._create_or_replace_popup, obj_name, forces)

    # ---------------- GUI ----------------
    def _gui_mainloop(self):
        """GUI-specific thread: Tk main loop + queue polling"""
        self._root = tk.Tk()
        self._root.withdraw()

        def poll_queue():
            try:
                while True:
                    fn, args = self._gui_queue.get_nowait()
                    fn(*args)
            except queue.Empty:
                pass
            self._root.after(50, poll_queue)

        self._root.after(0, poll_queue)
        self._root.mainloop()

    def _post_to_gui(self, fn, *args):
        self._gui_queue.put((fn, args))

    def _create_or_replace_popup(self, obj_name, forces):
        """GUI thread execution: display new popup"""
        # close old popup
        if self._current_popup and self._current_popup.winfo_exists():
            self._current_popup.destroy()

        popup = tk.Toplevel(self._root)
        popup.title("⚠️ Warning")
        popup.geometry("420x170")
        popup.resizable(False, False)
        try:
            popup.attributes("-topmost", True)
        except Exception:
            pass
        popup.lift()

        # force display: take first 3 components, pad with 0 if necessary
        fvals = list(forces[:3]) + [0.0] * (3 - len(forces))
        message = (
            f"{obj_name} has a large contact force!\n\n"
            f"Force value: [{fvals[0]:.3f}, {fvals[1]:.3f}, {fvals[2]:.3f}]\n"
            "It may damage the tool or fixture!"
        )

        label = tk.Label(popup, text=message, font=("Arial", 11),
                         fg="red", justify="center", wraplength=380)
        label.pack(expand=True, padx=12, pady=12)

        popup.after(1500, popup.destroy)  # 1.5s later, auto-close
        self._current_popup = popup

