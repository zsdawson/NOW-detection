#!/usr/bin/env python3

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import ImageTk

from config import MODEL_PATH, OUTPUT_DIR, PREVIEW_SIZE, CONFIDENCE_THRESHOLD
from camera_capture import PiCameraCapture
from model_runner import LocalYOLOModel
from storage_manager import StorageManager


class PiVisionKiosk:
    """
    Full-screen interactive kiosk app for Raspberry Pi 5.

    Main workflow:
    1. Live camera preview.
    2. Take Image.
    3. Run local YOLO model.
    4. Show annotated result.
    5. Save original image, annotated image, CSV, and metadata.
    6. Transfer saved files later through Wi-Fi download page.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Pi Vision Kiosk")
        self.root.configure(bg="#111111")

        # Makes the app fill the screen on boot.
        self.root.attributes("-fullscreen", True)

        # Main backend systems.
        self.camera = PiCameraCapture(preview_size=PREVIEW_SIZE)
        self.model = LocalYOLOModel(
            MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        self.storage = StorageManager(OUTPUT_DIR)

        # Current workflow state.
        self.current_image = None
        self.current_annotated_image = None
        self.current_detections = []
        self.current_model_error = None
        self.current_preview_photo = None
        self.preview_running = False

        # User-editable fields.
        self.session_var = tk.StringVar(value="Site1_Trap1")
        self.image_name_var = tk.StringVar(value="capture")
        self.output_dir_var = tk.StringVar(value=str(OUTPUT_DIR))
        self.status_var = tk.StringVar(value="Starting camera...")

        self._build_ui()
        self._bind_keys()

        # Start camera shortly after UI loads.
        self.root.after(200, self.start_camera)

    def _build_ui(self):
        """
        Builds the visible touchscreen/monitor interface.
        """

        # Top area: camera preview / captured image / annotated image.
        top = tk.Frame(self.root, bg="#111111")
        top.pack(fill="both", expand=True)

        self.image_label = tk.Label(
            top,
            text="Camera loading...",
            bg="#000000",
            fg="#ffffff",
            font=("Arial", 24),
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Bottom control panel.
        controls = tk.Frame(self.root, bg="#222222")
        controls.pack(fill="x", side="bottom", padx=10, pady=10)

        # Input row.
        row1 = tk.Frame(controls, bg="#222222")
        row1.pack(fill="x", pady=4)

        tk.Label(
            row1,
            text="Session:",
            bg="#222222",
            fg="#ffffff",
            font=("Arial", 14),
        ).pack(side="left")

        tk.Entry(
            row1,
            textvariable=self.session_var,
            font=("Arial", 14),
            width=20,
        ).pack(side="left", padx=8)

        tk.Label(
            row1,
            text="Image Name:",
            bg="#222222",
            fg="#ffffff",
            font=("Arial", 14),
        ).pack(side="left")

        tk.Entry(
            row1,
            textvariable=self.image_name_var,
            font=("Arial", 14),
            width=20,
        ).pack(side="left", padx=8)

        tk.Label(
            row1,
            text="Output:",
            bg="#222222",
            fg="#ffffff",
            font=("Arial", 14),
        ).pack(side="left")

        tk.Entry(
            row1,
            textvariable=self.output_dir_var,
            font=("Arial", 12),
            width=34,
        ).pack(side="left", padx=8)

        tk.Button(
            row1,
            text="Choose Folder",
            command=self.choose_folder,
            font=("Arial", 12),
        ).pack(side="left", padx=4)

        # Button row.
        row2 = tk.Frame(controls, bg="#222222")
        row2.pack(fill="x", pady=8)

        button_style = {
            "font": ("Arial", 16),
            "height": 2,
            "width": 13,
        }

        tk.Button(
            row2,
            text="Take Image",
            command=self.take_image,
            **button_style,
        ).pack(side="left", padx=6)

        tk.Button(
            row2,
            text="Run Model",
            command=self.run_model,
            **button_style,
        ).pack(side="left", padx=6)

        tk.Button(
            row2,
            text="Save Session",
            command=self.save_session,
            **button_style,
        ).pack(side="left", padx=6)

        tk.Button(
            row2,
            text="Transfer Mode",
            command=self.transfer_mode_info,
            **button_style,
        ).pack(side="left", padx=6)

        tk.Button(
            row2,
            text="Exit",
            command=self.exit_app,
            **button_style,
        ).pack(side="right", padx=6)

        # Status bar.
        status = tk.Label(
            controls,
            textvariable=self.status_var,
            bg="#222222",
            fg="#ffffff",
            anchor="w",
            font=("Arial", 12),
        )
        status.pack(fill="x", pady=4)

    def _bind_keys(self):
        """
        Keyboard shortcuts.

        ESC = exit
        F11 = toggle fullscreen
        Space = take image
        """

        self.root.bind("<Escape>", lambda event: self.exit_app())
        self.root.bind("<F11>", lambda event: self.toggle_fullscreen())
        self.root.bind("<space>", lambda event: self.take_image())

    def start_camera(self):
        """
        Starts the Raspberry Pi camera.
        """

        try:
            self.camera.start()
            self.preview_running = True
            self.status_var.set("Camera ready. Press Take Image.")
            self.update_preview()

        except Exception as exc:
            self.status_var.set(f"Camera error: {exc}")
            messagebox.showerror("Camera Error", str(exc))

    def update_preview(self):
        """
        Updates the live preview.

        It only shows the live camera feed when no image has been captured yet.
        Once an image is captured, the screen stays on that captured image until
        another workflow action changes it.
        """

        if not self.preview_running:
            return

        try:
            if self.current_image is None:
                frame = self.camera.get_preview_frame()
                self._show_image(frame)

        except Exception as exc:
            self.status_var.set(f"Preview error: {exc}")

        # Refresh preview roughly every 150 milliseconds.
        self.root.after(150, self.update_preview)

    def _show_image(self, image):
        """
        Scales a PIL image to fit the screen and displays it.
        """

        if image is None:
            return

        label_width = max(self.image_label.winfo_width(), 800)
        label_height = max(self.image_label.winfo_height(), 500)

        display = image.copy()
        display.thumbnail((label_width, label_height))

        self.current_preview_photo = ImageTk.PhotoImage(display)
        self.image_label.configure(
            image=self.current_preview_photo,
            text="",
        )

    def choose_folder(self):
        """
        Lets the user choose where saved sessions should go.
        """

        chosen = filedialog.askdirectory(initialdir=str(OUTPUT_DIR))

        if chosen:
            self.output_dir_var.set(chosen)
            self.storage = StorageManager(chosen)
            self.status_var.set(f"Output folder set to: {chosen}")

    def take_image(self):
        """
        Captures an image from the camera but does not save it yet.

        This allows the workflow:
        Take Image → Run Model → Save Session
        """

        try:
            self.current_image = self.camera.capture_unsaved_image()
            self.current_annotated_image = None
            self.current_detections = []
            self.current_model_error = None

            self._show_image(self.current_image)

            self.status_var.set(
                "Image captured. Press Run Model or Save Session."
            )

        except Exception as exc:
            self.status_var.set(f"Capture error: {exc}")
            messagebox.showerror("Capture Error", str(exc))

    def run_model(self):
        """
        Runs the local YOLO model on the current image.
        """

        if self.current_image is None:
            self.status_var.set("Take an image before running the model.")
            return

        try:
            self.status_var.set("Running model...")
            self.root.update_idletasks()

            result = self.model.run(self.current_image)

            self.current_annotated_image = result["annotated_image"]
            self.current_detections = result["detections"]
            self.current_model_error = result["error"]

            self._show_image(self.current_annotated_image)

            if result["error"]:
                self.status_var.set(
                    f"Model skipped/error: {result['error']}"
                )
            else:
                self.status_var.set(
                    f"Model complete. Detections: {result['total_detections']}"
                )

        except Exception as exc:
            self.status_var.set(f"Model error: {exc}")
            messagebox.showerror("Model Error", str(exc))

    def save_session(self):
        """
        Saves the current workflow.

        Saves:
        - original image
        - annotated image, if model was run
        - results.csv
        - metadata JSON
        """

        if self.current_image is None:
            self.status_var.set("No image to save. Press Take Image first.")
            return

        try:
            self.storage = StorageManager(self.output_dir_var.get())

            result = self.storage.save_capture_bundle(
                session_name=self.session_var.get(),
                image_name=self.image_name_var.get(),
                original_image=self.current_image,
                annotated_image=self.current_annotated_image,
                detections=self.current_detections,
                model_name=MODEL_PATH.name,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                model_error=self.current_model_error,
            )

            self.status_var.set(
                f"Saved. Session: {result['session_dir'].name}. "
                f"Detections: {result['total_detections']}"
            )

        except Exception as exc:
            self.status_var.set(f"Save error: {exc}")
            messagebox.showerror("Save Error", str(exc))

    def transfer_mode_info(self):
        """
        Shows transfer instructions.

        The actual transfer server is separate and should be running through
        transfer_server.py or the transfer systemd service.
        """

        messagebox.showinfo(
            "Transfer Mode",
            "Connect phone/laptop to the Pi hotspot, then open:\n\n"
            "http://10.42.0.1:5000\n\n"
            "If using normal Wi-Fi, run scripts/show_ip.sh and open:\n\n"
            "http://PI_IP:5000",
        )

    def toggle_fullscreen(self):
        """
        Toggles fullscreen mode.
        """

        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def exit_app(self):
        """
        Safely closes the kiosk app.
        """

        self.preview_running = False

        try:
            self.camera.stop()
        except Exception:
            pass

        self.root.destroy()


def main():
    root = tk.Tk()
    PiVisionKiosk(root)
    root.mainloop()


if __name__ == "__main__":
    main()
