from nltk.stem import WordNetLemmatizer
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext, Menu
import openai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
import shutil
from PIL import Image, ImageTk

# import os



def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            data: dict = json.load(file)
        return data
    except FileNotFoundError:
        messagebox.showerror("Error", "Knowledge base file not found.")
        return {"questions": []}

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def create_backup(file_path: str, backup_path: str):
    try:
        shutil.copyfile(file_path, backup_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create backup: {e}")

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def find_best_match_tfidf(user_question, questions):
    preprocessed_questions = [preprocess_text(q) for q in questions]
    preprocessed_user_question = preprocess_text(user_question)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_questions + [preprocessed_user_question])

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_index = cosine_similarities.argsort()[0][-1]

    if cosine_similarities[0][best_match_index] < 0.4:
        return None
    return questions[best_match_index]

def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content'].strip()

def toggle_fullscreen():
    if root.attributes('-fullscreen'):
        root.attributes('-fullscreen', False)
        chat_history.grid(row=0, column=0, columnspan=2, sticky="nsew")
        user_input_entry.grid(row=1, column=0, sticky="ew")
        send_button.grid(row=1, column=1, sticky="ew")
        fullscreen_button.config(text='Fullscreen')
    else:
        root.attributes('-fullscreen', True)
        chat_history.grid(row=0, column=0, columnspan=2, sticky="nsew")
        user_input_entry.grid(row=1, column=0, sticky="ew")
        send_button.grid(row=1, column=1, sticky="ew")
        fullscreen_button.config(text='Exit Fullscreen')

def send_message():
    global knowledge_base
    user_input = user_input_entry.get()
    user_input_entry.delete(0, tk.END)

    if user_input.lower() == 'quit':
        root.destroy()
        return

    update_chat("You", user_input)

    global use_gpt
    if use_gpt:
        response = chat_with_gpt(user_input)
        update_chat("Bot", response)
    else:
        best_match = find_best_match_tfidf(user_input, [q["question"] for q in knowledge_base["questions"]])
        if best_match:
            answer = get_answer_for_question(best_match, knowledge_base)
            update_chat("Bot", answer)
        else:
            update_chat("Bot", "I don't know the answer. Can you provide me the answer?")
            new_answer = simpledialog.askstring("Input", "Type the answer or 'skip' to skip:")
            if new_answer and new_answer.lower() != 'skip':
                knowledge_base["questions"].append({"question": user_input, "answer": new_answer})
                update_chat("Bot", "Thank you! I learned a new response!")
                save_knowledge_base('knowledge_base.json', knowledge_base)
            else:
                update_chat("Bot", "Okay, let's move on.")

def update_chat(user, text):
    chat_history.config(state='normal')
    chat_history.insert(tk.END, f"{user}: {text}\n")
    chat_history.yview(tk.END)
    chat_history.config(state='disabled')

def clear_chat():
    chat_history.config(state='normal')
    chat_history.delete('1.0', tk.END)
    chat_history.config(state='disabled')

def save_chat():
    with open('chat_history.txt', 'w') as file:
        chat_content = chat_history.get('1.0', tk.END)
        file.write(chat_content)

def open_file():
    try:
        with open('chat_history.txt', 'r') as file:
            chat_content = file.read()
            chat_history.config(state='normal')
            chat_history.delete('1.0', tk.END)
            chat_history.insert(tk.END, chat_content)
            chat_history.config(state='disabled')
    except FileNotFoundError:
        messagebox.showerror("Error", "File not found.")

knowledge_base = load_knowledge_base('knowledge_base.json')

backup_path = 'knowledge_base_backup.json'
create_backup('knowledge_base.json', backup_path)

root = tk.Tk()
root.title("AI Mahbub")
root.config(bg="Black")
icon_image = Image.open("rob.png")
icon_photo = ImageTk.PhotoImage(icon_image)
root.iconphoto(True, icon_photo)

font_style = ("Times New Roman", 14)
text_color = "white"

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)
root.grid_rowconfigure(2, weight=0)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=0)

chat_history = scrolledtext.ScrolledText(root, state='disabled', font=font_style, fg=text_color, bg="black")
chat_history.grid(row=0, column=0, columnspan=2, sticky="nsew")

update_chat("Bot", "Hello! I am your chatbot. How can I assist you today?")

user_input_entry = tk.Entry(root, font=font_style, fg=text_color, bg="grey")
user_input_entry.grid(row=1, column=0, sticky="ew")
send_button = tk.Button(root, text="Send", command=send_message, font=font_style, fg=text_color, bg="Black")
send_button.grid(row=1, column=1, sticky="ew")

use_gpt = False

fullscreen_button = tk.Button(root, text='Fullscreen', command=toggle_fullscreen, font=font_style, fg=text_color, bg="Green")
fullscreen_button.grid(row=2, column=0, columnspan=2, sticky="ew")

def switch_mode():
    global use_gpt
    use_gpt = not use_gpt
    update_chat("Bot", f"Switched to {'GPT' if use_gpt else 'Pre-trained'} mode")

def on_click(button):
    button.config(relief=tk.SUNKEN)
    root.after(200, lambda: button.config(relief=tk.RAISED))

send_button.bind("<Button-1>", lambda _: on_click(send_button))
fullscreen_button.bind("<Button-1>", lambda _: on_click(fullscreen_button))

menu_bar = Menu(root)
root.config(menu=menu_bar)

file_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

root.mainloop()
