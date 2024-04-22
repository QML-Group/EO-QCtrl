import tkinter as tk

def function():
    print("Button Test")

window = tk.Tk()
btn = tk.Button(window, text = "Button test widget", fg = "blue", command = function)
btn.place(x=80, y=100)
window.title("Test GUI")
window.geometry("300x200+10+10")
window.mainloop()
