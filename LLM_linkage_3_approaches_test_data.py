import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load models based on selection
def load_model(model_name):
    print(f"🔄 Loading model: {model_name}")
    if model_name == "Phi-2":
        model_id = "microsoft/phi-2"
        return pipeline("text-generation", model=model_id, device_map="auto", torch_dtype=torch.float16)
    elif model_name == "Mistral-7B (quantized)":
        model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    elif model_name == "OpenChat 3.5 (quantized)":
        model_id = "TheBloke/openchat-3.5-GPTQ"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt logic
def create_prompt(name1, dept1, school1, name2, dept2, school2):
    return (
        f"Do these two entries likely refer to the same person?\n\n"
        f"Entry A:\nName: {name1}\nDepartment: {dept1}\nSchool: {school1}\n\n"
        f"Entry B:\nName: {name2}\nDepartment: {dept2}\nSchool: {school2}\n\n"
        f"Reason through this and say Yes or No. If yes, why?"
    )

# Perform linking
def run_linking():
    file1 = file1_var.get()
    file2 = file2_var.get()
    model_name = model_var.get()
    if not file1 or not file2 or not model_name:
        messagebox.showerror("Error", "Please select both files and a model.")
        return

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    model = load_model(model_name)

    results = []
    for _, row1 in df1.iterrows():
        for _, row2 in df2.iterrows():
            prompt = create_prompt(
                row1["Name"], row1["Department"], row1["School"],
                row2["Name"], row2["Department"], row2["School"]
            )
            output = model(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
            results.append({
                "Name1": row1["Name"], "Name2": row2["Name"],
                "Prompt": prompt, "Model Output": output
            })

    out_path = "linked_name_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    messagebox.showinfo("Done", f"✅ Linking complete. Results saved to: {out_path}")

# GUI Setup
root = tk.Tk()
root.title("LLM Name Linker")

tk.Label(root, text="Select Dataset 1").grid(row=0, column=0, padx=5, pady=5)
tk.Label(root, text="Select Dataset 2").grid(row=1, column=0, padx=5, pady=5)
tk.Label(root, text="Choose Model").grid(row=2, column=0, padx=5, pady=5)

file1_var = tk.StringVar()
file2_var = tk.StringVar()
model_var = tk.StringVar()

tk.Entry(root, textvariable=file1_var, width=40).grid(row=0, column=1)
tk.Entry(root, textvariable=file2_var, width=40).grid(row=1, column=1)

def browse_file(var):
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        var.set(path)

tk.Button(root, text="Browse", command=lambda: browse_file(file1_var)).grid(row=0, column=2)
tk.Button(root, text="Browse", command=lambda: browse_file(file2_var)).grid(row=1, column=2)

model_select = ttk.Combobox(root, textvariable=model_var, values=[
    "Phi-2", "Mistral-7B (quantized)", "OpenChat 3.5 (quantized)"
])
model_select.grid(row=2, column=1)

tk.Button(root, text="Run Linking", command=run_linking, bg="#4CAF50", fg="white", padx=10).grid(row=3, column=1, pady=20)

root.mainloop()
