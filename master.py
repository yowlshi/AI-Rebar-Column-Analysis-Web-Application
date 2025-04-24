#!/usr/bin/env python3
# Optimized Rebar Analysis Application for Raspberry Pi 5 (800x480 display)

import os
import time
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox
import picamera2
import threading
import json
import math
import csv
import traceback
import gc  # Garbage collector
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

class RebarAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rebar Analysis")
        
        # Make it fullscreen for the 800x480 display
        self.root.attributes('-fullscreen', True)
        
        # Initialize variables
        self.captured_frame = None
        self.camera_paused = False
        self.current_results = []
        self.is_processing = False  # Flag to prevent multiple captures
        
        # Store image references to prevent premature garbage collection
        self.image_references = []
        
        # Define colors (accessible before loading models)
        self.colors = {
            'primary': '#3498db',      # Blue
            'accent': '#27ae60',       # Green
            'warning': '#e74c3c',      # Red
            'bg': '#f5f5f5',           # Light background
            'dark': '#2c3e50',         # Dark blue/gray
            'text': '#34495e',         # Dark text
            'light_text': '#ffffff'    # Light text
        }
        
        # Create results directory if it doesn't exist
        self.results_dir = "analysis_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Setup basic UI first so we can show loading status
        self.setup_basic_ui()
        
        # Load models and cement ratios
        try:
            self.load_models()
            self.load_cement_ratios()
            self.update_status("Ready", "normal")
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "error")
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
            print(f"Initialization error: {e}")
            print(traceback.format_exc())
        
        # Complete UI setup
        self.setup_ui_complete()
        
        # Initialize camera and start preview
        try:
            self.initialize_camera()
            self.start_preview()
        except Exception as e:
            self.update_status(f"Camera Error: {str(e)}", "error")
            messagebox.showerror("Camera Error", f"Failed to initialize camera: {str(e)}")
            print(f"Camera error: {e}")
            print(traceback.format_exc())
        
        # Set up periodic garbage collection
        self.setup_garbage_collection()
    
    def setup_garbage_collection(self):
        """Setup periodic garbage collection to prevent memory leaks"""
        def run_gc():
            gc.collect()
            self.root.after(30000, run_gc)  # Run every 30 seconds
        
        # Start the periodic garbage collection
        self.root.after(30000, run_gc)
    
    def setup_basic_ui(self):
        """Setup initial UI to show loading status"""
        # Create title bar with standard tk widgets
        self.create_title_bar()
        
        # Main container
        self.main_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Loading label
        self.loading_label = tk.Label(
            self.main_container,
            text="Loading models, please wait...",
            font=("Arial", 14),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        self.loading_label.pack(expand=True)
        
        # Status indicator at bottom
        self.status_frame = tk.Frame(self.main_container, bg=self.colors['bg'])
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_indicator = tk.Label(
            self.status_frame, 
            text="?", 
            font=("Arial", 12), 
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="Initializing...",
            font=("Arial", 10),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X)
        
        # Update the UI
        self.root.update()
    
    def update_status(self, message, status_type="normal"):
        """Update status message and indicator"""
        self.status_label.config(text=message)
        
        if status_type == "normal":
            color = self.colors['primary']
        elif status_type == "processing":
            color = self.colors['warning']
        elif status_type == "success":
            color = self.colors['accent']
        elif status_type == "error":
            color = self.colors['warning']
        else:
            color = self.colors['primary']
            
        self.status_indicator.config(fg=color)
        self.root.update()
    
    def create_title_bar(self):
        """Create a simple custom title bar using standard tk widgets"""
        # Create the title bar frame with standard tk.Frame
        self.title_bar = tk.Frame(self.root, bg=self.colors['dark'])
        self.title_bar.pack(fill=tk.X)
        
        # Add title with standard tk.Label
        title_label = tk.Label(
            self.title_bar, 
            text="Rebar Analysis", 
            font=('Arial', 10, 'bold'),
            bg=self.colors['dark'],
            fg=self.colors['light_text']
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Add exit button with standard tk.Button
        exit_btn = tk.Button(
            self.title_bar, 
            text="X", 
            width=3,
            font=('Arial', 8),
            bg=self.colors['dark'],
            fg=self.colors['light_text'],
            relief=tk.FLAT,
            command=self.quit_app
        )
        exit_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Add minimize button with standard tk.Button
        minimize_btn = tk.Button(
            self.title_bar, 
            text="-", 
            width=3,
            font=('Arial', 8),
            bg=self.colors['dark'],
            fg=self.colors['light_text'],
            relief=tk.FLAT,
            command=self.minimize_window
        )
        minimize_btn.pack(side=tk.RIGHT, padx=0, pady=5)
        
        # Make the title bar draggable
        title_label.bind("<ButtonPress-1>", self.start_move)
        title_label.bind("<ButtonRelease-1>", self.stop_move)
        title_label.bind("<B1-Motion>", self.do_move)
        
        # Store the position
        self.x = 0
        self.y = 0
    
    def quit_app(self):
        """Clean up resources before quitting"""
        try:
            # Stop the camera if it's running
            if hasattr(self, 'camera'):
                self.camera.stop()
                print("Camera stopped")
            
            # Clear image references
            self.image_references.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Quit the application
            print("Exiting application...")
            self.root.quit()
        except Exception as e:
            print(f"Error during exit: {e}")
            self.root.quit()
    
    def load_models(self):
        """Load the models using Detectron2"""
        self.update_status("Loading rebar detection model...", "processing")
        
        # Set device explicitly to CPU
        self.device = "cpu"
        
        # Rebar detection model
        self.rebar_cfg = get_cfg()
        self.rebar_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        self.rebar_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.rebar_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.rebar_cfg.MODEL.DEVICE = "cpu"
        self.rebar_model = build_model(self.rebar_cfg)
        DetectionCheckpointer(self.rebar_model).load("rebar_model1.pth")
        self.rebar_model.eval()
        
        self.update_status("Loading section detection model...", "processing")
        
        # Section detection model
        self.section_cfg = get_cfg()
        self.section_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        self.section_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.section_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.section_cfg.MODEL.DEVICE = "cpu"
        self.section_model = build_model(self.section_cfg)
        DetectionCheckpointer(self.section_model).load("section_model1.pth") 
        self.section_model.eval()
        
        print("Models loaded successfully")
    
    def load_cement_ratios(self):
        """Load cement mixture ratios based on rebar diameter"""
        # Default ratios if file doesn't exist
        self.cement_ratios = {
            "small": {"cement": 1, "sand": 2, "aggregate": 3, "diameter_range": [6, 12]},
            "medium": {"cement": 1, "sand": 2, "aggregate": 4, "diameter_range": [12, 20]},
            "large": {"cement": 1, "sand": 3, "aggregate": 5, "diameter_range": [20, 50]}
        }
        
        # Try to load from file
        try:
            if os.path.exists('cement_ratios.json'):
                with open('cement_ratios.json', 'r') as f:
                    self.cement_ratios = json.load(f)
        except Exception as e:
            print(f"Error loading cement ratios: {e}")
    
    def setup_ui_complete(self):
        """Setup the complete UI after models are loaded"""
        # Clear loading message
        for widget in self.main_container.winfo_children():
            if widget != self.status_frame:  # Keep the status frame
                widget.destroy()
        
        # Create a horizontal layout
        self.main_pane = tk.PanedWindow(self.main_container, orient=tk.HORIZONTAL, 
                                       bg=self.colors['bg'], sashwidth=4)
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for camera (65% width)
        self.left_panel = tk.Frame(self.main_pane, bg=self.colors['bg'])
        
        # Right panel for controls and results (35% width)
        self.right_panel = tk.Frame(self.main_pane, bg=self.colors['bg'])
        
        # Add panels to paned window with weights
        self.main_pane.add(self.left_panel, stretch="always", width=520)  # ~65% of 800px
        self.main_pane.add(self.right_panel, stretch="always", width=270)  # ~35% of 800px
        
        # Camera frame in left panel
        self.camera_frame = tk.LabelFrame(
            self.left_panel, 
            text="Camera", 
            font=("Arial", 10, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera preview
        self.preview_container = tk.Frame(self.camera_frame, bg="black")
        self.preview_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.preview_label = tk.Label(self.preview_container, bg="black")
        self.preview_label.pack(expand=True, anchor=tk.CENTER)
        
        # Controls in right panel
        self.controls_frame = tk.Frame(self.right_panel, bg=self.colors['bg'])
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Capture button with more reliable standard tk Button
        self.capture_btn = tk.Button(
            self.controls_frame, 
            text="Capture & Analyze", 
            font=("Arial", 11, "bold"),
            bg=self.colors['primary'],
            fg=self.colors['light_text'],
            activebackground=self.colors['primary'],
            activeforeground=self.colors['light_text'],
            relief=tk.RAISED,
            padx=10,
            pady=10,
            command=self.capture_image
        )
        self.capture_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Results frame
        self.results_frame = tk.LabelFrame(
            self.right_panel, 
            text="Results", 
            font=("Arial", 10, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Results text with standard Text widget
        self.results_text = tk.Text(
            self.results_frame, 
            wrap=tk.WORD, 
            font=("Arial", 11),
            bg=self.colors['bg'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=5,
            pady=5,
            height=12
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Add a scrollbar to the results text - standard Scrollbar is more reliable on RPi
        scrollbar = tk.Scrollbar(self.results_text, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Initial message
        self.results_text.insert(tk.END, "Press 'Capture & Analyze' to begin.\n")
        self.results_text.config(state=tk.DISABLED)
    
    def minimize_window(self):
        """Minimize the window"""
        self.root.attributes('-fullscreen', False)
        self.root.state('iconic')
    
    def start_move(self, event):
        """Start window movement"""
        self.x = event.x
        self.y = event.y
    
    def stop_move(self, event):
        """Stop window movement"""
        self.x = None
        self.y = None
    
    def do_move(self, event):
        """Move the window"""
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")
    
    def initialize_camera(self):
        """Initialize the camera with original dimensions"""
        try:
            if hasattr(self, 'camera'):
                # Clean up existing camera instance if it exists
                try:
                    self.camera.stop()
                except:
                    pass
            
            self.camera = picamera2.Picamera2()
            # Use original 300x400 resolution from the first code
            config = self.camera.create_preview_configuration(main={"size": (300, 400)})
            self.camera.configure(config)
            print("Camera initialized with original dimensions (300x400)")
            self.update_status("Camera initialized", "normal")
        except Exception as e:
            raise Exception(f"Failed to initialize camera: {e}")
    
    def start_preview(self):
        """Start camera preview"""
        try:
            self.camera.start()
            self.update_preview()
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            # Try to restart the camera
            self.restart_camera()
    
    def restart_camera(self):
        """Attempt to restart the camera if there's an issue"""
        try:
            print("Attempting to restart camera...")
            self.update_status("Restarting camera...", "processing")
            
            # Clean up existing camera if possible
            if hasattr(self, 'camera'):
                try:
                    self.camera.stop()
                except:
                    pass
            
            # Reinitialize the camera
            time.sleep(1)  # Wait a bit before restarting
            self.initialize_camera()
            self.camera.start()
            self.update_preview()
            
            self.update_status("Camera restarted", "success")
            print("Camera restarted successfully")
        except Exception as e:
            print(f"Failed to restart camera: {e}")
            self.update_status("Camera restart failed", "error")
    
    def update_preview(self):
        """Update the camera preview - keeping original dimensions"""
        # Skip updates if we're showing a result or processing
        if (hasattr(self, 'camera_paused') and self.camera_paused) or self.is_processing:
            self.root.after(500, self.update_preview)
            return
        
        try:
            frame = self.camera.capture_array()
            
            # Convert to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage without resizing to maintain original dimensions
            photo = ImageTk.PhotoImage(image=img)
            
            # Store a reference to prevent garbage collection
            if len(self.image_references) > 10:  # Keep only last 10 images
                self.image_references.pop(0)
            self.image_references.append(photo)
            
            # Update label with the latest image
            self.preview_label.config(image=photo)
            
            # Use 100ms update interval for better performance
            self.root.after(100, self.update_preview)
        except Exception as e:
            print(f"Error in update_preview: {e}")
            # Try to recover
            self.root.after(1000, self.update_preview)
            
    def capture_image(self):
        """Capture an image and start analysis"""
        # Check if already processing
        if self.is_processing:
            print("Already processing, ignoring capture request")
            return
            
        # Set processing flag
        self.is_processing = True
        
        # Disable capture button temporarily
        self.capture_btn.config(state=tk.DISABLED)
        self.update_status("Capturing image...", "processing")
        
        # Clear results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Capturing image...\n")
        self.results_text.config(state=tk.DISABLED)
        
        # Start capture in a separate thread
        threading.Thread(target=self._do_capture).start()
    
    def _do_capture(self):
        """Capture the image in a separate thread with original dimensions"""
        try:
            # Reset current results data
            self.current_results = []
            
            # Capture high-resolution image
            self.camera.stop()
            # Use original resolution from the first code
            config = self.camera.create_still_configuration(main={"size": (300, 400)})
            self.camera.configure(config)
            self.camera.start()
            time.sleep(0.5)
            
            # Capture image
            self.captured_frame = self.camera.capture_array()
            
            # Reset camera to preview mode
            self.camera.stop()
            # Go back to preview resolution
            config = self.camera.create_preview_configuration(main={"size": (300, 400)})
            self.camera.configure(config)
            self.camera.start()
            
            # Generate timestamp for this analysis session
            self.current_timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Create a unique folder for this analysis session
            self.current_result_dir = os.path.join(self.results_dir, f"analysis_{self.current_timestamp}")
            os.makedirs(self.current_result_dir, exist_ok=True)
            
            # Save original image in the analysis folder
            original_filename = os.path.join(self.current_result_dir, 'original_image.jpg')
            cv2.imwrite(original_filename, self.captured_frame)
            
            # Update results text
            self.update_results("Image captured. Starting analysis...\n")
            self.update_status("Analyzing image...", "processing")
            
            # Automatically start analysis
            self.root.after(0, self._do_analyze)
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            print(traceback.format_exc())
            self.root.after(0, lambda: self.update_status(f"Capture error: {str(e)}", "error"))
            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
            self.update_results(f"Error capturing image: {e}\n")
            self.is_processing = False  # Reset processing flag
            
            # Try to restart the camera if there was an error
            self.restart_camera()
    
    def _do_analyze(self):
        """Analyze the captured image"""
        try:
            if self.captured_frame is None:
                self.update_results("No image captured yet! Try again.\n")
                self.update_status("Error: No image", "error")
                self.capture_btn.config(state=tk.NORMAL)
                self.is_processing = False  # Reset processing flag
                return
            
            # Process with rebar model
            self.detect_rebar(self.captured_frame)
            
            # Update status
            self.root.after(0, lambda: self.update_status("Analysis complete", "success"))
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            print(traceback.format_exc())
            self.update_results(f"Error analyzing image: {e}\n")
            self.root.after(0, lambda: self.update_status("Analysis error", "error"))
            
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
            self.is_processing = False  # Reset processing flag
            
            # Force garbage collection
            gc.collect()
    
    def detect_rebar(self, frame):
        """First detect if there is a rebar in the image"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        height, width = frame_rgb.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            print(f"Resized image for analysis: {width}x{height} -> {new_width}x{new_height}")
        
        # Preprocess for model
        height, width = frame_rgb.shape[:2]
        image = torch.as_tensor(frame_rgb.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        # Run rebar detection model
        with torch.no_grad():
            outputs = self.rebar_model([inputs])[0]
        
        # Check if any rebars were detected
        if len(outputs["instances"]) == 0:
            self.update_results("No rebar detected in the image!\n")
            
            no_rebar_filename = os.path.join(self.current_result_dir, 'no_rebar_detected.jpg')
            cv2.imwrite(no_rebar_filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # Display the original image in camera panel
            self.display_result_in_camera_panel(frame_rgb)
            
            return
        
        # Get the highest-scoring rebar detection
        instances = outputs["instances"].to("cpu")
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        
        # Get the best rebar detection
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_box = boxes[best_idx].astype(int)
        
        
        # Draw the detected rebar
        rebar_image = frame_rgb.copy()
        x1, y1, x2, y2 = best_box
        cv2.rectangle(rebar_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rebar_image, f"Rebar: {best_score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save rebar detection result
        rebar_filename = os.path.join(self.current_result_dir, 'rebar_detected.jpg')
        cv2.imwrite(rebar_filename, cv2.cvtColor(rebar_image, cv2.COLOR_RGB2BGR))
        
        # Now detect sections within the detected rebar region
        self.detect_sections(frame_rgb, best_box)
    
    def detect_sections(self, frame_rgb, rebar_box):
        """Detect rebar sections within the detected rebar"""
        # Preprocess for model
        height, width = frame_rgb.shape[:2]
        image = torch.as_tensor(frame_rgb.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        
        # Run section detection model
        with torch.no_grad():
            outputs = self.section_model([inputs])[0]
        
        # Get the section instances
        instances = outputs["instances"].to("cpu")
        
        if len(instances) == 0:
            self.update_results("No rebar sections detected!\n")
            self.display_result_in_camera_panel(frame_rgb)
            return
        
        # Get detection details
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        # Create a result image
        result_image = frame_rgb.copy()
        
        # Generate colors for each section
        section_colors = []
        for i in range(len(boxes)):
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(100, 255)
            )
            section_colors.append(color)
        
        # Process each detected section
        self.update_results(f"Found {len(boxes)} sections\n")
        
        # List to store text results for each section
        section_text_results = []
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate diameter in pixels
            width_px = x2 - x1
            height_px = y2 - y1
            diameter_px = min(width_px, height_px)
            
            # Convert to real-world diameter in mm
            mm_per_pixel = 0.1  # Placeholder value
            diameter_mm = diameter_px * mm_per_pixel
            
            # Determine section size based on diameter
            size = "small"
            for category, info in self.cement_ratios.items():
                if "diameter_range" in info:
                    min_diam, max_diam = info["diameter_range"]
                    if min_diam <= diameter_mm < max_diam:
                        size = category
                        break
            
            # Get cement mixture ratio
            ratio = self.cement_ratios[size]
            
            # Create text result for this section
            section_result = {
                "section_id": i + 1,
                "size_category": size,
                "diameter_mm": round(diameter_mm, 2),
                "confidence": round(score, 3),
                "cement_ratio": ratio["cement"],
                "sand_ratio": ratio["sand"],
                "aggregate_ratio": ratio["aggregate"]
            }
            section_text_results.append(section_result)
            
            # Save section data for CSV
            section_data = {
                "timestamp": self.current_timestamp,
                **section_result
            }
            self.current_results.append(section_data)
            
            # Draw on the result image
            color = section_colors[i]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"S{i+1}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw mask if available
            if masks is not None:
                mask = masks[i]
                mask_colored = np.zeros_like(frame_rgb)
                mask_colored[mask] = color
                
                # Blend the mask with the image
                alpha = 0.4
                mask_region = mask.astype(bool)
                result_image[mask_region] = (
                    result_image[mask_region] * (1 - alpha) + 
                    mask_colored[mask_region] * alpha
                ).astype(np.uint8)
            
            # Update results
            self.update_results(f"Section {i+1} ({size}):\n")
            self.update_results(f"  Diam: {diameter_mm:.1f}mm\n")
            self.update_results(f"  Mix: C:{ratio['cement']}, S:{ratio['sand']}, A:{ratio['aggregate']}\n\n")
        
        # Save result image
        result_filename = os.path.join(self.current_result_dir, 'section_result.jpg')
        cv2.imwrite(result_filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Save analysis results to CSV
        self.save_results_to_csv()
        
        # Display result image in the camera panel
        self.display_result_in_camera_panel(result_image)
    
    def save_results_to_csv(self):
        """Save the current analysis results to a CSV file"""
        if not self.current_results:
            print("No results to save")
            return
        
        try:
            # Create a CSV file in the analysis folder
            filename = os.path.join(self.current_result_dir, 'analysis_data.csv')
            
            # Define CSV headers
            headers = [
                "timestamp", "section_id", "size_category", "diameter_mm", 
                "confidence", "cement_ratio", "sand_ratio", "aggregate_ratio"
            ]
            
            # Write data to CSV
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for result in self.current_results:
                    writer.writerow(result)
            
            print(f"Analysis data saved to {filename}")
            
            # Also save a summary text file
            summary_filename = os.path.join(self.current_result_dir, 'summary.txt')
            with open(summary_filename, 'w') as f:
                analysis_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(time.mktime(
                                                time.strptime(self.current_timestamp, '%Y%m%d-%H%M%S'))))
                f.write(f"Analysis: {analysis_time}\n\n")
                
                if not self.current_results:
                    f.write("No rebar sections detected.\n")
                else:
                    f.write(f"Found {len(self.current_results)} rebar sections:\n\n")
                    for result in self.current_results:
                        f.write(f"Section {result['section_id']} ({result['size_category']}):\n")
                        f.write(f"  Diameter: {result['diameter_mm']:.1f}mm\n")
                        f.write(f"  Mix: C:{result['cement_ratio']} S:{result['sand_ratio']} A:{result['aggregate_ratio']}\n\n")
            
            print(f"Summary saved to {summary_filename}")
            
            # Update status to show save happened
            self.update_status(f"Results saved to: analysis_{self.current_timestamp}", "success")
                
        except Exception as e:
            print(f"Error saving analysis data: {e}")
            print(traceback.format_exc())
    
    def display_result_in_camera_panel(self, image):
        """Display the result image in the camera panel"""
        try:
            # Convert to PIL Image
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=img)
            
            # Store reference to prevent garbage collection
            if len(self.image_references) > 10:
                self.image_references.pop(0)
            self.image_references.append(photo)
            
            # Update camera panel with result image
            self.preview_label.config(image=photo)
            
            # Set flag to stop camera preview
            self.camera_paused = True
            
            # Change the capture button to "Return to Camera"
            self.capture_btn.config(text="Return to Camera", command=self.resume_camera_preview)
        except Exception as e:
            print(f"Error displaying result: {e}")
            print(traceback.format_exc())
            
            # Try to resume camera preview if display fails
            self.resume_camera_preview()
    
    def resume_camera_preview(self):
        """Resume the camera preview after showing results"""
        # Set flag to resume camera preview
        self.camera_paused = False
        
        # Change the button back to "Capture & Analyze"
        self.capture_btn.config(text="Capture & Analyze", command=self.capture_image)
        self.update_status("Ready", "normal")
        
        # Clear any result image reference
        if hasattr(self, 'result_photo'):
            del self.result_photo
        
        # Force garbage collection
        gc.collect()
        
        # Restart camera preview updates
        self.update_preview()
    
    def update_results(self, text):
        """Update results text"""
        def _update():
            try:
                self.results_text.config(state=tk.NORMAL)
                self.results_text.insert(tk.END, text)
                self.results_text.see(tk.END)
                self.results_text.config(state=tk.DISABLED)
            except Exception as e:
                print(f"Error updating results: {e}")
        
        self.root.after(0, _update)

# Create default cement ratios file
def create_cement_ratios_file():
    """Create default cement ratios file"""
    ratios = {
        "small": {
            "cement": 1, 
            "sand": 2, 
            "aggregate": 3,
            "diameter_range": [6, 12]
        },
        "medium": {
            "cement": 1, 
            "sand": 2, 
            "aggregate": 4,
            "diameter_range": [12, 20]
        },
        "large": {
            "cement": 1, 
            "sand": 3, 
            "aggregate": 5,
            "diameter_range": [20, 50]
        }
    }
    
    with open('cement_ratios.json', 'w') as f:
        json.dump(ratios, f, indent=2)
    
    print("Created default cement ratios file")

# Main function
def main():
    # Create default ratios file if it doesn't exist
    if not os.path.exists('cement_ratios.json'):
        create_cement_ratios_file()
    
    # Error handling for the entire application
    try:
        # Create main window
        root = tk.Tk()
        
        # Create app
        app = RebarAnalysisApp(root)
        
        # Run the app
        root.mainloop()
    except Exception as e:
        print(f"Critical error: {e}")
        print(traceback.format_exc())
        
        # Try to show error in GUI if possible
        try:
            messagebox.showerror("Critical Error", f"Application failed to start: {str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()
