import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from ttkthemes import ThemedTk, ThemedStyle

from main import process_images, process_webcam, process_video

class AgeDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Age Detection Program")

        # Set the theme using ThemedStyle
        self.style = ThemedStyle(self.master)
        self.style.set_theme("plastik")  # Choose a theme that you like

        # Custom font and colors
        font_style = ("Arial", 14, "bold")  # Adjust the font size and style
        button_padx = 20  # Adjust padding as needed

        self.create_widgets()
        self.selected_model_path = None  # Variable to store the selected model path

    def create_widgets(self):
        # Create buttons with pack for center alignment

        self.webcam_button = tk.Button(
            self.master,
            text="Process Webcam",
            command=self.process_webcam,
            font=("Arial", 14, "bold"),
            padx=20,
            pady=10,
        )
        self.webcam_button.pack(pady=20)

        self.video_button = tk.Button(
            self.master,
            text="Process Video",
            command=self.process_video,
            font=("Arial", 14, "bold"),
            padx=20,
            pady=10,
        )
        self.video_button.pack(pady=20)

        self.image_button = tk.Button(
            self.master,
            text="Process Images",
            command=self.process_images,
            font=("Arial", 14, "bold"),
            padx=20,
            pady=10,
        )
        self.image_button.pack(pady=20)

        self.age_detection_model_button = tk.Button(
            self.master,
            text="Select Age Detection Model",
            command=self.select_model,
            font=("Arial", 14, "bold"),
            padx=20,
            pady=10,
        )
        self.age_detection_model_button.pack(pady=20)

    def process_webcam(self):
        process_webcam()

    def process_video(self):
        video_input_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")]
        )
        if video_input_path:
            video_output_path = filedialog.asksaveasfilename(
                title="Save Processed Video As",
                defaultextension=".mp4",
                filetypes=[("Video files", "*.mp4")],
            )
            if video_output_path:
                messagebox.showinfo("Information", "Processing Video...")  # Placeholder for feedback
                process_video(video_input_path, video_output_path)

    def process_images(self):
        # Ask for the input directory
        input_dir = filedialog.askdirectory(title="Select Image Directory")

        if not input_dir:
            return  # User canceled the operation

        # Ask for the output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return  # User canceled the operation

        messagebox.showinfo("Information", "Processing Images...")  # Placeholder for feedback
        process_images(input_dir, output_dir)

    def select_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model", filetypes=[("Model files", "*.h5;*.hdf5")]
        )
        if model_path:
            self.selected_model_path = model_path
            messagebox.showinfo("Information", "Model selected successfully.")
        else:
            messagebox.showerror("Error", "No model selected.")


def main():
    root = ThemedTk()  # Use ThemedTk for the main window
    app = AgeDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

