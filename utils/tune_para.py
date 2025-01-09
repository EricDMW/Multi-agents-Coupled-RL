#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : tune_para.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/utils)
@Date   : 01/08/2025
@Time   : 20:01:30
@Info   : Description of the script
"""

import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import json
import os

class ParameterAdjuster:
    @staticmethod
    def adjust_parameters_with_gui(parser, save_path=None):
        def save_parameters():
            parameters = {}
            for key, entry in entries.items():
                value = entry.get()
                if args_defaults[key].type:
                    try:
                        value = args_defaults[key].type(value)
                    except (ValueError, TypeError) as e:
                        messagebox.showerror("Invalid Input", f"Invalid value for {key}: {e}")
                        return
                args_defaults[key].default = value
                parameters[key] = value

            # Ensure save_path is a folder
            save_folder = save_path if save_path else "."
            os.makedirs(save_folder, exist_ok=True)

            # Find the smallest unused k
            existing_files = [f for f in os.listdir(save_folder) if f.endswith("_parameters.json")]
            existing_ks = [int(f.split("_")[0]) for f in existing_files if f.split("_")[0].isdigit()]
            k = 0
            if existing_ks:
                k = max(existing_ks) + 1
            
            output_file = os.path.join(save_folder, f"{k}_parameters.json")

            with open(output_file, 'w') as f:
                json.dump(parameters, f, indent=4)
            messagebox.showinfo("Success", f"Parameters saved to {output_file}")

            # Close the GUI
            root.destroy()
            root.quit()

        # Get the defaults and types from the parser
        args_defaults = {action.dest: action for action in parser._actions if action.dest != 'help'}

        # Create the main window
        root = tk.Tk()
        root.title("Parameter Adjustment GUI")

        # Create a scrollable frame
        frame = ttk.Frame(root)
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        frame.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create entries for each parameter
        entries = {}
        for i, (key, action) in enumerate(args_defaults.items()):
            ttk.Label(scrollable_frame, text=key).grid(row=i, column=0, padx=10, pady=5)
            entry = ttk.Entry(scrollable_frame)
            entry.insert(0, str(action.default))
            entry.grid(row=i, column=1, padx=10, pady=5)
            entries[key] = entry

        # Save button
        ttk.Button(root, text="Save", command=save_parameters).pack(pady=10)

        # Run the GUI loop
        root.mainloop()

        return parser  # Return the updated parser

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example parameter adjustment.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    # Adjust parameters and get the updated parser
    updated_parser = ParameterAdjuster.adjust_parameters_with_gui(parser, save_path="output_params")
    print("Final Parser Defaults:")
    for action in updated_parser._actions:
        if action.dest != 'help':
            print(f"{action.dest}: {action.default}")