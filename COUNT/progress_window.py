"""
A window that shows COUNT's progress while it counts.

When COUNT is run from PyCharm or a terminal, progress is also printed there as before. The COUNT app has no
console, so this window is the only place progress and errors appear.

The counting runs in a background thread; everything it prints (including the tqdm progress bars, which
rewrite their line with "\\r") is passed to the window through a queue, because tkinter may only be used from
the main thread.
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox

from COUNT import paths

POLL_MS = 100


class _QueueWriter:
    """A stand-in for sys.stdout / sys.stderr that sends text to the window and to the original stream."""

    def __init__(self, messages: queue.Queue, original):
        self.messages = messages
        self.original = original

    def write(self, text):
        if text:
            self.messages.put(text)
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def open_folder(path):
    """Open a folder in Explorer / Finder / the file manager."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: only on Windows
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class ProgressWindow:
    def __init__(self, master: tk.Tk, title: str, results_folder: str = ""):
        self.master = master
        self.results_folder = results_folder
        self.messages = queue.Queue()
        self.done = False
        self.error = None

        master.title(title)
        master.geometry("820x420")
        master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status = tk.StringVar(value="Counting...")
        self.status_label = tk.Label(master, textvariable=self.status, anchor="w", justify="left",
                                     wraplength=780, font=("TkDefaultFont", 11, "bold"))
        self.status_label.pack(side="top", fill="x", padx=10, pady=(8, 2))

        buttons = tk.Frame(master)
        buttons.pack(side="bottom", fill="x", padx=10, pady=8)
        self.close_button = tk.Button(buttons, text="Close", width=12, command=self.on_close)
        self.close_button.pack(side="right")
        self.open_button = tk.Button(buttons, text="Open results folder", command=self.open_results,
                                     state="disabled")
        self.open_button.pack(side="right", padx=8)

        self.text = tk.Text(master, wrap="none", height=18, font="TkFixedFont")
        scrollbar = tk.Scrollbar(master, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set, state="disabled")
        scrollbar.pack(side="right", fill="y", padx=(0, 10))
        self.text.pack(side="left", fill="both", expand=True, padx=(10, 0))

    def append(self, text):
        """Add printed text to the log. A "\\r" starts the current line again, as a terminal does for tqdm bars."""
        self.text.configure(state="normal")
        parts = text.split("\r")
        for i, part in enumerate(parts):
            if i > 0:
                self.text.delete("end-1c linestart", "end-1c")
            self.text.insert("end", part)
        self.text.configure(state="disabled")
        self.text.see("end")

    def poll(self):
        finished = self.done  # read before emptying the queue, so nothing printed at the end is missed
        pending = []
        try:
            while True:
                pending.append(self.messages.get_nowait())
        except queue.Empty:
            pass
        if pending:
            self.append("".join(pending))
        if finished:
            self.finish()
        else:
            self.master.after(POLL_MS, self.poll)

    def finish(self):
        if self.error is None:
            self.status.set("Finished." + (f" Results are in {self.results_folder}" if self.results_folder else ""))
        else:
            self.status.set(f"Finished with errors: {self.error}")
            self.status_label.configure(fg="firebrick")
        if self.results_folder and os.path.isdir(self.results_folder):
            self.open_button.configure(state="normal")

    def open_results(self):
        open_folder(self.results_folder)

    def on_close(self):
        if not self.done:
            if not messagebox.askyesno("Stop counting?", "COUNT is still counting. Stop and close COUNT?",
                                       parent=self.master):
                return
            os._exit(1)  # the counting thread and its worker processes can't be stopped more gently
        self.master.destroy()


def run_with_progress(work, title="COUNT", results_folder=""):
    """
    Run work() in a background thread while a progress window shows everything it prints.
    Returns when the user closes the window after work() finishes.
    """
    root = tk.Tk()
    try:
        root.iconbitmap(paths.ICON_PATH)
    except tk.TclError:
        pass  # .ico icons are only supported on Windows
    window = ProgressWindow(root, title, results_folder)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _QueueWriter(window.messages, original_stdout)
    sys.stderr = _QueueWriter(window.messages, original_stderr)

    def target():
        try:
            work()
        except BaseException as e:  # shown in the window rather than lost
            if type(e).__name__ != "CountingFailed":  # its reasons have already been printed
                import traceback
                traceback.print_exc()
            window.error = e
        finally:
            window.done = True

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    root.after(POLL_MS, window.poll)
    try:
        root.mainloop()
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
    return window
