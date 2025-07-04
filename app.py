# Importing the required libraries
import os
import string
import random
import re
import logging
import warnings
import time 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load distilgpt2 from local directory
LOCAL_MODEL_PATH = "./distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH)
model.eval()

app = Flask(__name__) #initializes the web server
CORS(app)  # Allow frontend to communicate with backend

# Suppress NLTK download output
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Suppress TensorFlow and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Text processing
lemmer = nltk.stem.WordNetLemmatizer()
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def lemtokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def lemnormalize(text):
    return lemtokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Preprocess text for TF-IDF Vectorizer
def preprocess_text(sentences):
    return [" ".join(lemnormalize(sentence)) for sentence in sentences]

# Load and parse the chatbot.txt data
def load_chatbot_data(filename='chatbot.txt'):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # Use encoding instead of ignoring errors
            raw_doc = f.read().lower()
            qa_pairs = raw_doc.split('\n\n')  # Split each Q&A pair by double newlines
            qa_dict = {}
            for pair in qa_pairs:
                if 'q:' in pair and 'a:' in pair:
                    question_answer = pair.split('a:', 1)  # Ensure only the first occurrence is split
                    question = question_answer[0].replace('q:', '').strip()  # Remove 'Q:' and strip whitespace
                    answer = question_answer[1].strip()  # Strip whitespace from the answer
                    qa_dict[question] = answer
            logging.info(f"Loaded {len(qa_dict)} question-answer pairs from '{filename}'.")
            return qa_dict, raw_doc
    except FileNotFoundError:
        logging.error(f"Error: The file '{filename}' was not found.")
        return {}, ""
    except Exception as e:
        logging.error(f"An error occurred while loading chatbot data: {e}")
        return {}, ""
    
def generate_gpt2_response(prompt, max_length=50):
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 50256  # The typical EOS token ID for GPT2 models
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Create attention mask
    attention_mask = inputs.ne(tokenizer.eos_token_id).long()

    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        attention_mask=attention_mask
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_response = response[len(prompt):].strip()

    # Basic quality checks
    if not clean_response or len(clean_response.split()) <= 2:
        return "I'm not sure how to respond to that. Could you rephrase?"
    return clean_response

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

# Load the chatbot data
qa_dict, raw_doc = load_chatbot_data()

# Greeting function
GREET_INPUTS = ('hello', 'hi', 'greetings', 'sup', "what's up", 'hey', 'good morning', 'good afternoon', 'good evening')
GREET_RESPONSES = [
    'Hello! How can I assist you today?',
    'Hi there! What can I help you with?',
    'Hey! I’m here to assist. What’s up?',
    'Greetings! How’s it going?'
]

def greet(sentence):
    sentence = sentence.lower()
    for greeting in GREET_INPUTS:
        if greeting in sentence:
            return random.choice(GREET_RESPONSES)
    return None

# Casual conversation responses
CONVERSATIONAL_INPUTS = {
    "i am fine": ["Glad to hear that! How can I assist you further?", "Great! How can I help you today?"],
    "how are you": ["I'm just a bot, but thanks for asking! How can I assist you today?",
                    "I'm doing well! How about you?"],
    "great": ["That's awesome! How can I assist you further?", "I'm glad to hear that! What's next?"],
    "good": ["That's great! What can I help you with?", "I'm happy to hear that! How can I assist you?"],
    "what's your name": ["I'm Guide, your assistant. What's your name?"],
    "what is your name": ["I'm Guide, your assistant. What's your name?"],  # Added for "What is your name?"
    "thank you": ["You're very welcome! Is there anything else I can help with?"],
    "cool": ["Nice! How can I assist you further?"]
}

# Improved casual conversation handler
def casual_conversation(user_response):
    user_response = user_response.lower()

    # Improved: Catch common name queries like "what is your name" or "who are you"
    if "your name" in user_response or "who are you" in user_response:
        return "I am Stark, your assistant. What is your name?"

    # Handle the predefined casual responses
    for phrase, responses in CONVERSATIONAL_INPUTS.items():
        if phrase in user_response:
            return random.choice(responses)
    return None

# Global variable to track user name
user_name = None

# Excluding common words from being recognized as names
common_words = GREET_INPUTS + tuple(CONVERSATIONAL_INPUTS.keys())

def name_introduction(user_response):
    global user_name  # Use the global variable to track username
    user_response = user_response.lower()

    # Skip if response is too short or common word
    if len(user_response.split()) <= 1 and user_response in common_words:
        return None

    # Refined regex patterns to capture the user's name in different phrases
    patterns = [
    r"my name is ([a-zA-Z]+(?:\s[a-zA-Z]+)*)", # "My name is Eben"
    r"i am ([a-zA-Z]+(?:\s[a-zA-Z]+)*)",# "I am Eben"
    r"name is ([a-zA-Z]+(?:\s[a-zA-Z]+)*)" # "Name is Eben"
]

    # First, check for explicit name introductions
    for pattern in patterns:
        match = re.search(pattern, user_response)
        if match:
            user_name = match.group(1).capitalize()  # Capitalize the name
            return f"Nice to meet you, {user_name}!"

    # If no pattern matches and the user input is a single word that is not a common phrase, treat it as a name
    if len(user_response.split()) == 1 and user_response.isalpha() and user_response not in common_words:
        user_name = user_response.capitalize()
        return f"Nice to meet you, {user_name}!"

    return None

# Finding answers in Q&A dictionary
def find_answer_in_qa_dict(user_response, qa_dict):
    user_response = user_response.lower().strip()
    best_match = None
    max_similarity = 0.3  # Minimum similarity threshold

    for question, answer in qa_dict.items():
        question_words = set(question.lower().split())
        user_words = set(user_response.split())

        match_score = len(user_words & question_words) / max(len(question_words), 1)
        if match_score > max_similarity:
            max_similarity = match_score
            best_match = answer

    return best_match if best_match else None

# Fallback responses
DEFAULT_RESPONSE = [
    "I'm not sure I understand, but let's try again!",
    "Hmmm, I'm still learning, but I’ll do my best to help!",
    "Can you rephrase that for me? I’m still figuring things out!",
    "I don’t have an answer for that right now, but I'll try to learn more!"
]
COURSE_NAVIGATION_RESPONSES = {
    "assignments": "You can find your assignments in the 'Assignments' section of the LMS.",
    "courses": "You can view your enrolled courses under the 'My Courses' section. ",
}

def course_navigation(user_response):
    user_response = user_response.lower()

    for key, response in COURSE_NAVIGATION_RESPONSES.items():
        if key in user_response:
            return response
    return None

def get_fallback_response():
    return random.choice(DEFAULT_RESPONSE)

# Main response function
# Preprocess the chatbot text into sentence tokens once
sent_tokens = nltk.sent_tokenize(raw_doc) if raw_doc else []

conversation_state = {
    "awaiting_name": False,
    "last_topic": None
}


def response(user_response):
    user_response = user_response.lower().strip()

    # Check for greeting first
    greeting_response = greet(user_response)
    if greeting_response:
        return greeting_response, False

    # Check if the user introduces their name
    name_response = name_introduction(user_response)
    if name_response:
        return name_response, False

    # Handle casual conversation
    casual_response = casual_conversation(user_response)
    if casual_response:
        return casual_response, False

    # Exit responses
    if user_response in ['no', 'nah', 'nope']:
        return "Thank you for chatting! Have a great day!", True
    elif user_response in ['yes', 'yeah', 'sure']:
        return "Great! What else would you like help with?", False

    # Check for predefined answers
    predefined_answer = find_answer_in_qa_dict(user_response, qa_dict)
    if predefined_answer:
        return predefined_answer.strip(), False

    # Similarity matching if no predefined answer is found
    if not sent_tokens:  # Ensure there are sentences to compare against
        return get_fallback_response(), False

    sent_tokens.append(user_response)
    preprocessed_sent_tokens = preprocess_text(sent_tokens)

    tfidvec = TfidfVectorizer(stop_words="english")
    tfidf = tfidvec.fit_transform(preprocessed_sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    sent_tokens.pop()  # Remove the user response after processing

    if predefined_answer:
        # Format answer with steps (replace numbers with newlines)
        formatted_answer = predefined_answer.strip()
        # Add newlines before numbered steps (1., 2., etc.)
        formatted_answer = re.sub(r'(\d+\.)', r'\n\1', formatted_answer)
        return formatted_answer, False

    if req_tfidf < 0.3:
        gpt2_reply = generate_gpt2_response(user_response)
        return f"Hmm... let me think: {gpt2_reply}", False
    else:
        return sent_tokens[idx], False

# Function to run interactive chatbot
def run_chatbot():
    flag = True
    print("BOT: Hi there! I am Stark, your friendly assistant. How can I help you today?")

    while flag:
        # Add a short sleep time to prevent blocking
        time.sleep(0.1)
        
        user_response = input("user: ").lower().strip()

        if not user_response:
            print("BOT: Can you say that again? I'm not sure I caught it.")
            continue

        bot_answer, exit_chat = response(user_response)
        print("BOT: " + bot_answer)
        logging.info(f"User input: {user_response} | Bot response: {bot_answer}")

        if exit_chat:
            print("BOT: Goodbye! Thank you for chatting!")
            logging.info("Chat ended by user.")
            flag = False

@app.route('/get-response', methods=['POST'])
def get_response():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "Invalid request. Please send a message field."}), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({"response": "Please enter a message."}), 400

        bot_answer, exit_chat = response(user_message)
        return jsonify({"response": bot_answer})
    
    except Exception as e:
        logging.error(f"Error in API request: {e}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
