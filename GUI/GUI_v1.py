import tkinter
import tkinter.messagebox
import customtkinter as ctk

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("green") 

class EUQOC_App(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Energy Efficient Universal Quantum Optimal Control")
        self.geometry(f"{1100}x{580}")

        # Configure grid layout
        self.grid_columnconfigure((0, 1, 2, 3), weight = 0)
        self.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight = 0)

        # Create Input Parameter Section
        self.inputparams_frame = ctk.CTkFrame(self)
        self.inputparams_frame.grid(row = 0, column = 0, padx=(20, 0), pady=(20, 0), sticky = "nsew")
        self.inputparams_label = ctk.CTkLabel(self.inputparams_frame, text = "Input Parameters", font = ctk.CTkFont(size = 20, weight = "bold"))
        self.inputparams_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))

        self.u_t_optionmenu = ctk.CTkOptionMenu(self.inputparams_frame, dynamic_resizing=False,
                                                values=["CNOT", "Hadamard", "T"])
        self.u_t_optionmenu.grid(row = 1, column = 0, padx = 20, pady = (20, 10))
        self.u_t_optionmenu.set("Target Unitary")

        self.h_d_optionmenu = ctk.CTkOptionMenu(self.inputparams_frame, dynamic_resizing=False,
                                                values=["HD1", "HD2"])
        self.h_d_optionmenu.grid(row = 1, column = 1, padx = 20, pady = (20, 10))
        self.h_d_optionmenu.set("Drift Hamiltonian")

        self.h_c_optionmenu = ctk.CTkOptionMenu(self.inputparams_frame, dynamic_resizing=False,
                                                values=["HC1", "HC2", "HC3", "HC4"])
        self.h_c_optionmenu.grid(row = 2, column = 0, padx = 20, pady = (20, 10))
        self.h_c_optionmenu.set("Control Hamiltonian")

        self.t_1_optionmenu = ctk.CTkEntry(self.inputparams_frame, placeholder_text = "Decoherence Time")
        self.t_1_optionmenu.grid(row = 2, column = 1, padx = 20, pady = (20, 10))

        self.N_g_optionmenu = ctk.CTkEntry(self.inputparams_frame, placeholder_text = "GRAPE Iterations")
        self.N_g_optionmenu.grid(row = 1, column = 2, padx = 20, pady = (20, 10))

        self.N_t_optionmenu = ctk.CTkEntry(self.inputparams_frame, placeholder_text = "Training Episodes")
        self.N_t_optionmenu.grid(row = 2, column = 2, padx = 20, pady = (20, 10))


        # Create Algorithm Selector 

        self.algoselect_frame = ctk.CTkFrame(self)
        self.algoselect_frame.grid(row = 1, column = 0, padx = (20, 0), pady = (20, 0), sticky = "nsew")
        self.algoselect_label = ctk.CTkLabel(self.algoselect_frame, text = "Algorithm Selector", font = ctk.CTkFont(size = 20, weight = "bold"))
        self.algoselect_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))

        self.eo_grape_switch = ctk.CTkSwitch(master = self.algoselect_frame, text = "EO-GRAPE")
        self.eo_grape_switch.grid(row = 1, column = 0, padx = 20, pady = (20, 10))

        self.rla1_switch = ctk.CTkSwitch(master = self.algoselect_frame, text = "RLA-1")
        self.rla1_switch.grid(row = 1, column = 1, padx = 20, pady = (20, 10))

        self.rla2_switch = ctk.CTkSwitch(master = self.algoselect_frame, text = "RLA-2")
        self.rla2_switch.grid(row = 1, column = 2, padx = 20, pady = (20, 10))

        # Create Weight Selector

        self.weightselect_frame = ctk.CTkFrame(self)
        self.weightselect_frame.grid(row = 2, column = 0, padx = (20, 0), pady = (20, 0), sticky = "nsew")
        self.weightselect_label = ctk.CTkLabel(self.weightselect_frame, text = "Weight Selector", font = ctk.CTkFont(size = 20, weight = "bold"))
        self.weightselect_label.grid(row = 0, column = 0, padx = 10, pady = (20, 10))
        self.w_e_label = ctk.CTkLabel(self.weightselect_frame, text = "Weight EC", font = ctk.CTkFont(size = 14))
        self.w_e_label.grid(row = 1, column = 0, padx = 10, pady = (20, 10))
        self.weightselect_slider_1 = ctk.CTkSlider(self.weightselect_frame, orientation = "horizontal", from_=0, to=1, number_of_steps=10, width = 300, progress_color="green", button_hover_color="darkgreen")
        self.weightselect_slider_1.grid(row = 1, column = 1)
        self.w_f_label = ctk.CTkLabel(self.weightselect_frame, text = "Weight F", font = ctk.CTkFont(size = 14))
        self.w_f_label.grid(row = 2, column = 0, padx = 10, pady = (20, 10))
        self.weightselect_slider_2 = ctk.CTkSlider(self.weightselect_frame, orientation = "horizontal", from_=0, to=1, number_of_steps=10, width = 300, progress_color="red", button_hover_color="darkred", button_color="red")
        self.weightselect_slider_2.grid(row = 2, column = 1)


if __name__ == "__main__":
    app = EUQOC_App()
    app.mainloop()