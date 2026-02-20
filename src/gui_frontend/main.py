"""
Main entry point for the GUI Frontend application.

This module initializes and launches the CustomTkinter-based GUI for visualizing
simulation results and controlling the autosim_core computational engine.
"""

import sys
from pathlib import Path

# Ensure the src directory is in the Python path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def main():
    """
    Main entry point for the GUI application.
    
    Initializes the CustomTkinter app and starts the main event loop.
    """
    try:
        import customtkinter as ctk
    except ImportError:
        print("Error: customtkinter is not installed.")
        print("Please install it with: pip install customtkinter")
        sys.exit(1)
    
    print("Starting TIFR-WORK GUI Application...")
    
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"
    
    # Create the main application window
    class TIFRWorkApp(ctk.CTk):
        """Main application window for TIFR-WORK GUI."""
        
        def __init__(self):
            super().__init__()
            
            # Configure window
            self.title("TIFR-WORK - Monte Carlo Simulation Suite")
            self.geometry("1200x800")
            
            # Configure grid layout
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)
            
            # Create main frame
            self.main_frame = ctk.CTkFrame(self)
            self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
            self.main_frame.grid_columnconfigure(0, weight=1)
            
            # Add title label
            self.title_label = ctk.CTkLabel(
                self.main_frame,
                text="TIFR-WORK Simulation Suite",
                font=ctk.CTkFont(size=24, weight="bold")
            )
            self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))
            
            # Add subtitle
            self.subtitle_label = ctk.CTkLabel(
                self.main_frame,
                text="Monte Carlo Simulation & Molecular Dynamics Visualization",
                font=ctk.CTkFont(size=14)
            )
            self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
            
            # Add placeholder content
            self.content_label = ctk.CTkLabel(
                self.main_frame,
                text="Welcome to TIFR-WORK!\n\n"
                     "This is the GUI frontend for:\n"
                     "• autosim_core: Monte Carlo simulation engine\n"
                     "• vmd_plugins: VMD integration tools\n\n"
                     "The application structure is ready for development.",
                font=ctk.CTkFont(size=12),
                justify="left"
            )
            self.content_label.grid(row=2, column=0, padx=20, pady=20)
            
            # Add a button
            self.button = ctk.CTkButton(
                self.main_frame,
                text="Start Simulation",
                command=self.start_simulation,
                font=ctk.CTkFont(size=14),
                height=40
            )
            self.button.grid(row=3, column=0, padx=20, pady=(10, 20))
        
        def start_simulation(self):
            """Placeholder for starting a simulation."""
            print("Simulation button clicked!")
            # TODO: Implement simulation start logic
            # This will interface with autosim_core package
    
    # Create and run the application
    app = TIFRWorkApp()
    app.mainloop()


if __name__ == "__main__":
    main()
