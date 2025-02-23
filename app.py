

# import tkinter as tk
# from tkinter import scrolledtext, messagebox
# import matplotlib.pyplot as plt
# import networkx as nx
# from collections import Counter
# from PIL import Image, ImageTk
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')

# def extract_key_phrases(text):
#     """Extracts meaningful key phrases using TF-IDF analysis."""
#     sentences = sent_tokenize(text)
    
#     # Handle empty input or single sentences
#     if not sentences or len(sentences) < 2:
#         return ["Main Topic"]
        
#     vectorizer = TfidfVectorizer(
#         stop_words='english',
#         ngram_range=(1, 3),
#         max_features=20
#     )
    
#     try:
#         tfidf_matrix = vectorizer.fit_transform(sentences)
#         feature_names = vectorizer.get_feature_names_out()
#         scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
#         phrase_scores = list(zip(feature_names, scores))
#         sorted_phrases = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
        
#         # Select top 8 key phrases
#         key_phrases = [phrase for phrase, score in sorted_phrases[:8]]
#         return key_phrases
#     except ValueError:
#         return ["Main Topic"]  # Fallback if TF-IDF fails

# def create_mind_map(text):
#     """Generates a structured mind map with a hierarchical structure."""
#     key_phrases = extract_key_phrases(text)
    
#     # Create hierarchical structure
#     tree = {"Main Topic": {}}
    
#     # Use first 3 phrases as main branches
#     main_branches = key_phrases[:3]
#     sub_branches = key_phrases[3:]
    
#     # Create the tree structure
#     for main in main_branches:
#         tree["Main Topic"][main] = {}
        
#     # Distribute remaining phrases as sub-branches
#     for i, sub in enumerate(sub_branches):
#         parent = main_branches[i % len(main_branches)]
#         tree["Main Topic"][parent][sub] = {}
    
#     return tree

# def draw_mind_map(tree, filename="mind_map.png"):
#     """Draws the mind map using NetworkX and Matplotlib."""
#     G = nx.Graph()
    
#     def add_nodes(parent, subtree, level=0):
#         G.add_node(parent, level=level)
#         for child in subtree:
#             G.add_node(child, level=level+1)
#             G.add_edge(parent, child)
#             add_nodes(child, subtree[child], level+1)
    
#     root = list(tree.keys())[0]
#     add_nodes(root, tree[root])
    
#     plt.figure(figsize=(16, 12))
#     plt.title("Mind Map", fontsize=16, pad=20)
    
#     # Use spring layout for node positioning
#     pos = nx.spring_layout(G, k=2, iterations=50)
    
#     # Define visual properties for different levels
#     node_sizes = []
#     node_colors = []
#     font_sizes = []
    
#     for node in G.nodes():
#         level = G.nodes[node]['level']
#         if level == 0:  # Root node
#             node_sizes.append(5000)
#             node_colors.append('#FF9999')
#             font_sizes.append(12)
#         elif level == 1:  # Main concepts
#             node_sizes.append(3500)
#             node_colors.append('#99CCFF')
#             font_sizes.append(10)
#         else:  # Sub concepts
#             node_sizes.append(2500)
#             node_colors.append('#99FF99')
#             font_sizes.append(9)
    
#     # Draw the graph
#     nx.draw(G, pos,
#             node_color=node_colors,
#             node_size=node_sizes,
#             font_size=10,
#             font_weight='bold',
#             edge_color='#666666',
#             width=2,
#             with_labels=True,
#             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
#     plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()

# class MindMapGenerator(tk.Tk):
#     def __init__(self):
#         super().__init__()
        
#         self.title("Smart Mind Map Generator")
#         self.configure(bg="#E6F3FF")
#         self.geometry("900x700")
        
#         self.create_widgets()
    
#     def create_widgets(self):
#         # Title Frame
#         title_frame = tk.Frame(self, bg="#E6F3FF")
#         title_frame.pack(pady=20)
        
#         title_label = tk.Label(
#             title_frame,
#             text="📚 Smart Mind Map Generator",
#             font=("Helvetica", 24, "bold"),
#             bg="#E6F3FF",
#             fg="#2E4053"
#         )
#         title_label.pack()
        
#         # Main Frame
#         main_frame = tk.Frame(self, bg="#E6F3FF")
#         main_frame.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)
        
#         # Instructions
#         instructions = tk.Label(
#             main_frame,
#             text="Paste your text below to generate a mind map",
#             font=("Helvetica", 12),
#             bg="#E6F3FF",
#             fg="#34495E"
#         )
#         instructions.pack(pady=(0, 10))
        
#         # Text Area
#         self.text_area = scrolledtext.ScrolledText(
#             main_frame,
#             wrap=tk.WORD,
#             width=70,
#             height=15,
#             font=("Helvetica", 12),
#             bg="white",
#             fg="#2C3E50"
#         )
#         self.text_area.pack(pady=10)
        
#         # Button Frame
#         button_frame = tk.Frame(main_frame, bg="#E6F3FF")
#         button_frame.pack(pady=20)
        
#         self.generate_button = tk.Button(
#             button_frame,
#             text="Generate Mind Map 🎯",
#             command=self.generate_mind_map,
#             font=("Helvetica", 12, "bold"),
#             bg="#3498DB",
#             fg="white",
#             padx=20,
#             pady=10
#         )
#         self.generate_button.pack(side=tk.LEFT, padx=10)
        
#         self.clear_button = tk.Button(
#             button_frame,
#             text="Clear Text 🗑️",
#             command=self.clear_text,
#             font=("Helvetica", 12),
#             bg="#E74C3C",
#             fg="white",
#             padx=20,
#             pady=10
#         )
#         self.clear_button.pack(side=tk.LEFT, padx=10)
    
#     def clear_text(self):
#         self.text_area.delete(1.0, tk.END)
    
#     def generate_mind_map(self):
#         text = self.text_area.get("1.0", tk.END).strip()
        
#         if not text:
#             messagebox.showwarning("Warning", "Please enter some text first!")
#             return
        
#         try:
#             mind_map_data = create_mind_map(text)
#             draw_mind_map(mind_map_data)
#             self.display_mind_map()
#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
#     def display_mind_map(self):
#         viewer = tk.Toplevel(self)
#         viewer.title("Mind Map Viewer")
#         viewer.geometry("1000x800")
#         viewer.configure(bg="#E6F3FF")
        
#         try:
#             img = Image.open("mind_map.png")
#             # Scale image to fit window while maintaining aspect ratio
#             display_size = (900, 700)
#             img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
#             photo = ImageTk.PhotoImage(img)
#             label = tk.Label(viewer, image=photo, bg="#E6F3FF")
#             label.image = photo  # Keep a reference!
#             label.pack(expand=True, pady=20)
            
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to load mind map: {str(e)}")

# if __name__ == "__main__":
#     app = MindMapGenerator()
#     app.mainloop()

import tkinter as tk
from tkinter import scrolledtext, messagebox
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from PIL import Image, ImageTk
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def extract_key_phrases(text):
    """Extracts meaningful key phrases using TF-IDF analysis and noun extraction."""
    sentences = sent_tokenize(text)
    
    # Fallback extraction for short texts
    def extract_fallback():
        words = word_tokenize(text.lower())
        tagged = nltk.pos_tag(words)
        nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']
                  and word not in stopwords.words('english')]
        if not nouns:
            return ["Main Topic"]
        counter = Counter(nouns)
        return [noun for noun, count in counter.most_common(8)]
    
    if not sentences:
        return extract_fallback()
    
    # Use TF-IDF for longer texts
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_features=50,
        token_pattern=r'(?u)\b[a-zA-Z-]{3,}\b'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        sorted_indices = np.argsort(-scores)
        key_phrases = [feature_names[i] for i in sorted_indices[:8]]
        return key_phrases
    except ValueError:
        return extract_fallback()

def create_mind_map(text):
    """Generates a hierarchical mind map structure from key phrases."""
    key_phrases = extract_key_phrases(text)
    
    if not key_phrases:
        return {"Main Topic": {}}
    
    # Extract main topic and branches
    main_topic = key_phrases[0].title()
    main_branches = list(dict.fromkeys(key_phrases[1:4]))  # Remove duplicates
    sub_branches = list(dict.fromkeys(key_phrases[4:8]))
    
    # Build mind map structure
    mind_map = {main_topic: {}}
    
    # Add main branches
    for branch in main_branches:
        mind_map[main_topic][branch] = {}
    
    # Add sub-branches with intelligent grouping
    for sub in sub_branches:
        added = False
        # Try to find parent branch through keyword matching
        for branch in main_branches:
            if branch.lower() in sub.lower() or sub.lower() in branch.lower():
                mind_map[main_topic][branch][sub] = {}
                added = True
                break
        # Fallback to round-robin assignment
        if not added and main_branches:
            parent = main_branches[len(mind_map[main_topic]) % len(main_branches)]
            mind_map[main_topic][parent][sub] = {}
    
    return mind_map

def draw_mind_map(tree, filename="mind_map.png"):
    """Visualizes the mind map using NetworkX with improved layout."""
    G = nx.DiGraph()  # Use directed graph for hierarchy
    
    def add_nodes(parent, subtree, level=0):
        # Set level attribute for the parent node
        G.add_node(parent, level=level)
        for child in subtree:
            # Add child node with level+1
            G.add_node(child, level=level+1)
            G.add_edge(parent, child)
            # Recursively add children with increased level
            add_nodes(child, subtree[child], level+1)
    
    root = list(tree.keys())[0]
    add_nodes(root, tree[root], 0)  # Start with root at level 0
    
    plt.figure(figsize=(16, 12))
    plt.title("Mind Map", fontsize=20, pad=20)
    
    # Use multipartite layout with the level attribute
    pos = nx.multipartite_layout(G, subset_key="level")
    
    # Visual settings based on level
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        level = G.nodes[node]['level']
        node_colors.append('#FF9999' if level == 0 else '#99CCFF' if level == 1 else '#99FF99')
        node_sizes.append(5000 if level == 0 else 3500 if level == 1 else 2500)
    
    nx.draw(G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=12,
            font_weight='bold',
            edge_color='#666666',
            width=2,
            with_labels=True,
            arrows=False,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

class MindMapGenerator(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Smart Mind Map Generator")
        self.configure(bg="#E6F3FF")
        self.geometry("900x700")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title Frame
        title_frame = tk.Frame(self, bg="#E6F3FF")
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="📚 Smart Mind Map Generator",
            font=("Helvetica", 24, "bold"),
            bg="#E6F3FF",
            fg="#2E4053"
        )
        title_label.pack()
        
        # Main Frame
        main_frame = tk.Frame(self, bg="#E6F3FF")
        main_frame.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="Paste your text below to generate a mind map",
            font=("Helvetica", 12),
            bg="#E6F3FF",
            fg="#34495E"
        )
        instructions.pack(pady=(0, 10))
        
        # Text Area
        self.text_area = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            width=70,
            height=15,
            font=("Helvetica", 12),
            bg="white",
            fg="#2C3E50"
        )
        self.text_area.pack(pady=10)
        
        # Button Frame
        button_frame = tk.Frame(main_frame, bg="#E6F3FF")
        button_frame.pack(pady=20)
        
        self.generate_button = tk.Button(
            button_frame,
            text="Generate Mind Map 🎯",
            command=self.generate_mind_map,
            font=("Helvetica", 12, "bold"),
            bg="#3498DB",
            fg="white",
            padx=20,
            pady=10
        )
        self.generate_button.pack(side=tk.LEFT, padx=10)
        
        self.clear_button = tk.Button(
            button_frame,
            text="Clear Text 🗑️",
            command=self.clear_text,
            font=("Helvetica", 12),
            bg="#E74C3C",
            fg="white",
            padx=20,
            pady=10
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
    
    def generate_mind_map(self):
        text = self.text_area.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text first!")
            return
        
        try:
            mind_map_data = create_mind_map(text)
            draw_mind_map(mind_map_data)
            self.display_mind_map()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def display_mind_map(self):
        viewer = tk.Toplevel(self)
        viewer.title("Mind Map Viewer")
        viewer.geometry("1000x800")
        viewer.configure(bg="#E6F3FF")
        
        try:
            img = Image.open("mind_map.png")
            # Scale image to fit window while maintaining aspect ratio
            display_size = (900, 700)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(viewer, image=photo, bg="#E6F3FF")
            label.image = photo  # Keep a reference!
            label.pack(expand=True, pady=20)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mind map: {str(e)}")

if __name__ == "__main__":
    app = MindMapGenerator()
    app.mainloop()