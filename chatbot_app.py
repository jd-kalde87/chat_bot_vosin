import customtkinter as ctk
from PIL import Image
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- CARGAR EL CEREBRO DEL CHATBOT (sin cambios) ---
lemmatizer = WordNetLemmatizer()
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- FUNCIONES PARA PROCESAR Y OBTENER RESPUESTA (sin cambios) ---
def process_text(text):
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return ' '.join(lemmatized_tokens)

def get_bot_response(user_text):
    processed_user_text = process_text(user_text)
    user_vector = vectorizer.transform([processed_user_text])
    predicted_tag = model.predict(user_vector)[0]
    for intent in intents['intenciones']:
        if intent['etiqueta'] == predicted_tag:
            return random.choice(intent['respuestas'])
    return "Lo siento, no te entendí bien. ¿Puedes preguntarme de otra manera?"

# --- NUEVA INTERFAZ GRÁFICA CON SOLUCIÓN AL ERROR ---

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Asistente Eléctrico")
        self.geometry("420x550")
        self.resizable(False, False)

        # --- CARGAR IMÁGENES ---
        try:
            self.logo_photo = ctk.CTkImage(Image.open("logo.png"), size=(100, 40))
            self.robot_photo = ctk.CTkImage(Image.open("robot_icon.png"), size=(30, 30))
        except FileNotFoundError:
            self.logo_photo = None
            self.robot_photo = None
            print("Advertencia: No se encontraron 'logo.png' o 'robot_icon.png'.")

        # --- LAYOUT PRINCIPAL ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- WIDGETS ---
        top_frame = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        if self.logo_photo:
            logo_label = ctk.CTkLabel(top_frame, image=self.logo_photo, text="")
            logo_label.pack(side="left", padx=(0, 10))

        title_label = ctk.CTkLabel(top_frame, text="Asistente Virtual", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(side="left")
        
        # **CAMBIO CLAVE**: Usamos un CTkScrollableFrame en lugar de CTkTextbox
        self.chat_log_frame = ctk.CTkScrollableFrame(self)
        self.chat_log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        bottom_frame = ctk.CTkFrame(self, height=70, corner_radius=0)
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.entry_box = ctk.CTkEntry(bottom_frame, placeholder_text="Escribe tu mensaje aquí...")
        self.entry_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.entry_box.bind("<Return>", self.send_message)

        self.send_button = ctk.CTkButton(bottom_frame, text="Enviar", width=70, command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky="nsew")

    def send_message(self, event=None):
        msg = self.entry_box.get().strip()
        if not msg:
            return
        self.entry_box.delete(0, "end")
        
        self.display_message(msg, "user")
        
        # Obtener respuesta del bot y mostrarla con un pequeño retraso
        self.after(500, self.bot_reply, msg)

    def bot_reply(self, user_msg):
        res = get_bot_response(user_msg)
        self.display_message(res, "bot")
    
    def display_message(self, msg, sender):
        # **CAMBIO CLAVE**: Creamos un Frame para cada mensaje
        msg_frame = ctk.CTkFrame(self.chat_log_frame, fg_color="transparent")
        
        if sender == "bot":
            if self.robot_photo:
                icon_label = ctk.CTkLabel(msg_frame, image=self.robot_photo, text="")
                icon_label.pack(side="left", padx=(0, 10))
            
            text_bubble = ctk.CTkLabel(msg_frame, text=msg, wraplength=250, justify="left", 
                                       fg_color="#4B4B4B", text_color="white", corner_radius=10, 
                                       font=ctk.CTkFont(size=13))
            text_bubble.pack(side="left", pady=5, ipady=5, ipadx=5)
            msg_frame.pack(side="top", anchor="w", fill="x", padx=5, pady=5)
            
        else: # Mensaje del usuario
            text_bubble = ctk.CTkLabel(msg_frame, text=msg, wraplength=250, justify="left", 
                                       fg_color="#0084ff", text_color="white", corner_radius=10, 
                                       font=ctk.CTkFont(size=13))
            text_bubble.pack(side="right", pady=5, ipady=5, ipadx=5)
            msg_frame.pack(side="top", anchor="e", fill="x", padx=5, pady=5)
        
        # Auto-scroll hacia el final
        self.after(100, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        self.chat_log_frame._parent_canvas.yview_moveto(1.0)

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()