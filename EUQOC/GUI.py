import tkinter
import tkinter.messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import environments as env
import qrla as qrla

"""
Work in progress
----------------

This file contains a GUI to initiate and run the EUQOC code
"""

class EUQOC_App(ctk.CTk):

    ctk.set_appearance_mode("dark")  
    ctk.set_default_color_theme("green") 

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Energy Efficient Universal Quantum Optimal Control")
        self.geometry(f"{1100}x{580}")

        # Configure grid layout
        self.grid_columnconfigure((0, 1), weight = 0)
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight = 0)

        # Create Input Parameter Section
        self.inputparams_frame = ctk.CTkFrame(self)
        self.inputparams_frame.grid(row = 0, column = 0, padx=(20, 0), pady=(20, 0), sticky = "nsew")
        self.inputparams_label = ctk.CTkLabel(self.inputparams_frame, text = "Input Parameters", font = ctk.CTkFont(size = 24, weight = "bold"))
        self.inputparams_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))

        self.u_t_optionmenu = ctk.CTkOptionMenu(self.inputparams_frame, dynamic_resizing=False,
                                                values=["CNOT", "Hadamard", "T"], fg_color="white", text_color="green")
        self.u_t_optionmenu.grid(row = 1, column = 0, padx = 20, pady = (20, 10))
        self.u_t_optionmenu.set("Target Unitary")

        self.h_d_optionmenu = ctk.CTkOptionMenu(self.inputparams_frame, dynamic_resizing=False,
                                                values=["HD1", "HD2"], fg_color="white", text_color="green")
        self.h_d_optionmenu.grid(row = 1, column = 1, padx = 20, pady = (20, 10))
        self.h_d_optionmenu.set("Drift Hamiltonian")

        self.h_c_optionmenu = ctk.CTkOptionMenu(self.inputparams_frame, dynamic_resizing=False,
                                                values=["HC1", "HC2", "HC3", "HC4"], fg_color="white", text_color="green")
        self.h_c_optionmenu.grid(row = 2, column = 0, padx = 20, pady = (20, 10))
        self.h_c_optionmenu.set("Control Hamiltonian")

        self.t_1_optionmenu = ctk.CTkEntry(self.inputparams_frame, placeholder_text = "Decoherence Time", fg_color="white", text_color="green", placeholder_text_color= "green")
        self.t_1_optionmenu.grid(row = 2, column = 1, padx = 20, pady = (20, 10))

        self.N_g_optionmenu = ctk.CTkEntry(self.inputparams_frame, placeholder_text = "GRAPE Iterations", fg_color="white", text_color="green", placeholder_text_color= "green")
        self.N_g_optionmenu.grid(row = 1, column = 2, padx = 20, pady = (20, 10))

        self.N_t_optionmenu = ctk.CTkEntry(self.inputparams_frame, placeholder_text = "Training Episodes", fg_color="white", text_color="green", placeholder_text_color= "green")
        self.N_t_optionmenu.grid(row = 2, column = 2, padx = 20, pady = (20, 10))


        # Create Algorithm Selector 
        self.algoselect_frame = ctk.CTkFrame(self)
        self.algoselect_frame.grid(row = 1, column = 0, padx = (20, 0), pady = (20, 0), sticky = "nsew")
        self.algoselect_label = ctk.CTkLabel(self.algoselect_frame, text = "Algorithm Selector", font = ctk.CTkFont(size = 24, weight = "bold"))
        self.algoselect_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))

        self.eo_grape_switch = ctk.CTkSwitch(master = self.algoselect_frame, text = "EO-GRAPE", command = self.buttonfunction)
        self.eo_grape_switch.grid(row = 1, column = 0, padx = 20, pady = (20, 10))

        self.rla1_switch = ctk.CTkSwitch(master = self.algoselect_frame, text = "RLA-1", command = self.buttonfunction)
        self.rla1_switch.grid(row = 1, column = 1, padx = 20, pady = (20, 10))

        self.rla2_switch = ctk.CTkSwitch(master = self.algoselect_frame, text = "RLA-2", command = self.buttonfunction)
        self.rla2_switch.grid(row = 1, column = 2, padx = 20, pady = (20, 10))

        # Create Weight Selector

        self.weightselect_frame = ctk.CTkFrame(self)
        self.weightselect_frame.grid(row = 2, column = 0, padx = (20, 0), pady = (20, 0), sticky = "nsew")
        self.weightselect_label = ctk.CTkLabel(self.weightselect_frame, text = f"Weight Selector", font = ctk.CTkFont(size = 24, weight = "bold"))
        self.weightselect_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))
        self.w_e_label = ctk.CTkLabel(self.weightselect_frame, text = "Weight EC", font = ctk.CTkFont(size = 14))
        self.w_e_label.grid(row = 1, column = 0, padx = 10, pady = (20, 10))
        self.weightselect_slider_1 = ctk.CTkSlider(self.weightselect_frame, orientation = "horizontal", from_=0, to=1, number_of_steps=10, width = 300, progress_color="green", button_hover_color="darkgreen")
        self.weightselect_slider_1.grid(row = 1, column = 1)
        self.w_f_label = ctk.CTkLabel(self.weightselect_frame, text = "Weight F", font = ctk.CTkFont(size = 14))
        self.w_f_label.grid(row = 2, column = 0, padx = 10, pady = (20, 10))
        self.weightselect_slider_2 = ctk.CTkSlider(self.weightselect_frame, orientation = "horizontal", from_=0, to=1, number_of_steps=10, width = 300, progress_color="red", button_hover_color="darkred", button_color="red")
        self.weightselect_slider_2.grid(row = 2, column = 1)

        # Run Button

        self.runbutton_frame = ctk.CTkFrame(self)
        self.runbutton_frame.grid(row = 3, column = 0, padx = (20, 0), pady = (20, 0), sticky = "nsew")
        self.runbutton = ctk.CTkButton(self.runbutton_frame, text="Run Experiment", width = 600)
        
        self.runbutton.grid(column = 1)

        # Results frame

        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row = 0, column = 1, padx = (20, 0), pady = (20, 0), sticky = "nsew", rowspan = 4)
        self.results_label = ctk.CTkLabel(self.results_frame, text = "Simulation Results", font = ctk.CTkFont(size = 24, weight = "bold"))
        self.results_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))

        # Matplotlib results 
        
        fig = Figure(figsize = (4, 3), dpi = 100)
        t = np.arange(0, 3, .01)
        fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

        fig_2 = Figure(figsize = (4, 3), dpi = 100)
        t = np.arange(0, 3, .01)
        fig_2.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

        canvas = FigureCanvasTkAgg(fig, master = self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 0, padx = 10, pady = (20, 10))

        canvas_2 = FigureCanvasTkAgg(fig, master = self.results_frame)
        canvas_2.draw()
        canvas_2.get_tk_widget().grid(row = 2, column = 0, padx = 10, pady = (20, 10))

    def buttonfunction(self):

        if self.eo_grape_switch.get() == 1:
            algorithm = "EO-GRAPE"

        elif self.rla1_switch.get() == 1:
            algorithm = "RLA-1"

        elif self.rla2_switch.get() == 1:
            algorithm = "RLA-2"

        else:
            algorithm = "None"

        print(algorithm)

if __name__ == "__main__":
    app = EUQOC_App()
    app.mainloop()

