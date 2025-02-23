import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from PIL import Image, ImageTk

# Download required NLTK data
nltk.download('punkt')

def extract_key_points(text):
    """Extracts key points from a paragraph by identifying common words."""
    sentences = sent_tokenize(text)
    word_freq = Counter(word_tokenize(text.lower()))

    # Ignore common words
    common_words = set(["is", "the", "a", "and", "to", "of", "it", "for", "on", "with", "in", "that", "as", "at", "by"])
    key_sentences = {}

    for sentence in sentences:
        words = [word for word in word_tokenize(sentence.lower()) if word.isalnum() and word not in common_words]
        importance = sum(word_freq[word] for word in words)
        key_sentences[sentence] = importance

    # Get top 5 important sentences
    sorted_sentences = sorted(key_sentences, key=key_sentences.get, reverse=True)[:5]
    return sorted_sentences

def create_mind_map(text):
    """Generates a structured mind map from extracted key points."""
    key_sentences = extract_key_points(text)
    tree = {"Mind Map": {f"Point {i+1}": {sentence: {}} for i, sentence in enumerate(key_sentences)}}
    return tree

def draw_mind_map(tree, filename="mind_map.png"):
    """Draws a mind map using NetworkX and Matplotlib and saves it as an image."""
    G = nx.DiGraph()

    def add_edges(parent, subtree):
        for child in subtree:
            G.add_edge(parent, child)
            add_edges(child, subtree[child])

    root = list(tree.keys())[0]
    add_edges(root, tree[root])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    plt.title("Generated Mind Map", fontsize=14, color='darkblue')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="#ffccd5", edge_color="gray", font_size=10)
    plt.savefig(filename)  # Save the mind map as an image
    plt.close()

def show_welcome_screen():
    """Displays a welcome screen for 3 seconds."""
    welcome = tk.Toplevel(root)
    welcome.configure(bg="#ffccd5")
    welcome.geometry("400x200")
    welcome_label = tk.Label(welcome, text="✨ Welcome to Mind Map Generator by Billy Billy ✨", font=("Arial", 14), bg="#ffccd5")
    welcome_label.pack(expand=True)
    welcome.after(3000, welcome.destroy)

def generate_mind_map():
    """Processes user input and generates a mind map."""
    user_input = text_area.get("1.0", tk.END).strip()

    if len(user_input.split()) > 500:
        result_label.config(text="❌ Error: Notes must be under 500 words!", fg="red")
        return

    show_loading_screen(user_input)

def show_loading_screen(user_input):
    """Shows a loading screen with funny prompts while processing."""
    loading_screen = tk.Toplevel(root)
    loading_screen.configure(bg="#ffccd5")
    loading_screen.geometry("400x200")
    loading_label = tk.Label(loading_screen, text="Generating your mind map...", font=("Arial", 14), bg="#ffccd5")
    loading_label.pack(pady=20)

    funny_prompts = [
        "Thinking deep thoughts... 🧠",
        "Untangling the mess in your mind... 🤯",
        "Summoning the mind map gods... ⛩",
        "Bribing neurons to connect... 🤝",
        "Turning coffee into code... ☕"
    ]

    for _ in range(5):  
        loading_label.config(text=random.choice(funny_prompts))
        loading_screen.update()
        time.sleep(1)

    loading_screen.destroy()
    mind_map_data = create_mind_map(user_input)
    draw_mind_map(mind_map_data, "mind_map.png")

    # Display the mind map image in the tkinter window
    display_mind_map("mind_map.png")

def display_mind_map(image_path):
    """Displays the generated mind map image in the tkinter window."""
    mind_map_window = tk.Toplevel(root)
    mind_map_window.title("Generated Mind Map")
    mind_map_window.geometry("800x600")

    img = Image.open(image_path)
    img = img.resize((750, 550), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)

    image_label = tk.Label(mind_map_window, image=img_tk)
    image_label.image = img_tk  # Keep a reference to avoid garbage collection
    image_label.pack()

def enable_paste(event):
    """Allows pasting using Ctrl+V, Shift+Insert, and right-click."""
    text_area.event_generate("<<Paste>>")
    return "break"

def enable_right_click_paste(event):
    """Allows right-click paste."""
    text_area.event_generate("<<Paste>>")
    return "break"

# Create UI window
root = tk.Tk()
root.withdraw()
show_welcome_screen()

time.sleep(3)

root.deiconify()
root.title("Mind Map Generator")
root.configure(bg="#ffccd5")

# Input Window
text_window = tk.Toplevel(root)
text_window.configure(bg="#ffccd5")
text_window.geometry("600x400")
text_window.title("Enter Your Notes")

text_label = tk.Label(text_window, text="Paste your text (Max 500 words):", font=("Arial", 12), bg="#ffccd5")
text_label.pack(pady=10)

# Improved Text Area (Now Supports Ctrl+V, Right-Click, and Shift+Insert)
text_area = scrolledtext.ScrolledText(text_window, wrap=tk.WORD, width=60, height=15, font=("Arial", 12))
text_area.pack(pady=10)

# Enable all paste methods
text_area.bind("<Control-v>", enable_paste)
text_area.bind("<Shift-Insert>", enable_paste)
text_area.bind("<Button-3>", enable_right_click_paste)

# Auto-focus text area to allow pasting
text_area.focus_set()

result_label = tk.Label(text_window, text="", font=("Arial", 10), bg="#ffccd5")
result_label.pack()

generate_button = tk.Button(text_window, text="Generate Mind Map", command=generate_mind_map, bg="#ff99aa", font=("Arial", 12))
generate_button.pack(pady=10, fill=tk.X)  # Fix button shrinking issue

root.mainloop()