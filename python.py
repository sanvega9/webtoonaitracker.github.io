import tkinter as tk
from tkinter import scrolledtext

# Bot response function
def get_bot_response(user_input):
    user_input = user_input.lower()
    if "hello" in user_input or "hi" in user_input:
        return "Hi there! How can I help you?"
    elif "how are you" in user_input:
        return "I'm just a bot, but I'm doing great!"
    elif "bye" in user_input:
        return "Goodbye! Talk to you later!"
    else:
        return "I'm not sure what you mean. Can you ask differently?"

# Function to send message
def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() != "":
        chat_display.config(state='normal')
        chat_display.insert(tk.END, "You: " + user_input + "\n")
        response = get_bot_response(user_input)
        chat_display.insert(tk.END, "Bot: " + response + "\n\n")
        chat_display.config(state='disabled')
        chat_display.yview(tk.END)
        entry.delete(0, tk.END)

# Create window
root = tk.Tk()
root.title("Simple Chatbot")
root.geometry("400x500")

# Chat display area
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 12))
chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Entry field
entry = tk.Entry(root, font=("Arial", 14))
entry.pack(padx=10, pady=5, fill=tk.X)
entry.bind("<Return>", send_message)  # Pressing Enter sends message

# Send button
send_btn = tk.Button(root, text="Send", font=("Arial", 12), command=send_message)
send_btn.pack(pady=5)

# Start GUI loop
root.mainloop()
