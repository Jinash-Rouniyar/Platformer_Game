from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
import json
from typing import Dict, Any, List
from enum import Enum
from db_setup import get_db_connection
import time
from datetime import datetime

class TaskType(Enum):
    REGISTER_CONTENT = "register_content"
    SUMMARY_ASSIGNMENT = "summary_assignment"
    CAUSAL_RELATION = "causal_relation_analysis"
    NONE = "none"
    INAPPROPRIATE = "inappropriate"

class ConversationHistory:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        return self.messages[-n:] if n > 0 else self.messages
    
    def get_conversation(self):
        return self.messages
    
    def clear(self):
        self.messages = []

class RegistrationBot:
    CONTENT_TYPES = [
        "Online textbook",
        "Hard copy",
        "Video lectures",
        "YouTube clips",
        "Presentation slides",
        "Blogs",
        "PDF copies"
    ]
    
    CONFIRMATION_KEYWORDS = [
        "done", "confirm", "uploaded", "yes", "completed", "finished", "great"
    ]
    
    def __init__(self, conversation_history: ConversationHistory):
        self.current_state = "ask_content_type"
        self.selected_content_type = None
        self.conversation_history = conversation_history
        self.content_to_register = None  # Store the content here
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        # Add user input to conversation history
        self.conversation_history.add_message("user", user_input)
        
        if self.current_state == "ask_content_type":
            result = self._handle_content_type_selection(user_input)
        elif self.current_state == "handle_upload":
            # Store potential content before checking for confirmation
            if not any(keyword in user_input.lower() for keyword in self.CONFIRMATION_KEYWORDS):
                self.content_to_register = user_input
            result = self._handle_upload_process(user_input)
        
        # Add bot response to conversation history
        self.conversation_history.add_message("assistant", result["text"])
        return result
    
    def _handle_content_type_selection(self, user_input: str) -> Dict[str, Any]:
        content_type = user_input.strip().lower()
        valid_types = [ct.lower() for ct in self.CONTENT_TYPES]
        
        if content_type not in valid_types:
            return {
                "text": f"Please select a valid content type from the following:\n" + 
                       "\n".join(self.CONTENT_TYPES),
                "task_success": False
            }
            
        self.selected_content_type = content_type
        self.current_state = "handle_upload"
        
        instructions = self._get_upload_instructions(content_type)
        return {
            "text": instructions,
            "task_success": False
        }
    
    def _get_upload_instructions(self, content_type: str) -> str:
        instructions = {
            "pdf copies": "Please click the upload button and select your PDF file.",
            "online textbook": "Please provide the URL of the online textbook.",
            "hard copy": "Please enter the ISBN and edition of the textbook.",
            "video lectures": "Please upload the video file or provide the secure URL.",
            "youtube clips": "Please provide the YouTube video URL.",
            "presentation slides": "Please upload your presentation files (PPT, PPTX, or PDF format).",
            "blogs": "Please provide the blog URL and author information."
        }
        return instructions.get(content_type, "Please upload your content.")
    
            
    def _handle_upload_process(self, user_input: str) -> Dict[str, Any]:
        # Check if any confirmation keyword is in the user input
        if any(keyword in user_input.lower() for keyword in self.CONFIRMATION_KEYWORDS):
            # Get last few messages to check context
            last_messages = self.conversation_history.get_last_n_messages(4)
            
            # Check if we've already seen an upload instruction
            upload_instruction_seen = any(
                "upload" in msg["content"].lower() 
                for msg in last_messages 
                if msg["role"] == "assistant"
            )
            
            # If we've seen the upload instruction and user confirms, accept it
            if upload_instruction_seen and self.content_to_register:
                return {
                    "text": f"Successfully registered {self.selected_content_type}!",
                    "task_success": True,
                    "content": self.content_to_register,  # Return the stored content
                    "type": self.selected_content_type
                }
        
        return {
            "text": "Just a quick check—can you verify and confirm if everything looks good on your end? Let me know if any adjustments are needed.",
            "task_success": False
        }

class EducationBot:
    def __init__(self, model_path: str):
        # Initialize BERT model for classification
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize ChatOpenAI
        self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.output_parser = StrOutputParser()
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory()
        
        # Initialize registration bot
        self.registration_bot = None
        self.current_task = None
        
    def classify_input(self, user_input: str) -> TaskType:
        prediction = self.nlp(user_input)
        return TaskType(prediction[0]["label"])
        
    def process_input(self, user_input: str) -> Dict[str, Any]:
        if self.registration_bot and not self.registration_bot.current_state == "completed":
            result = self.registration_bot.process_input(user_input)
            if result["task_success"]:
                self.registration_bot = None
                self.current_task = None
            return result
            
        # Add the initial user input to conversation history
        self.conversation_history.add_message("user", user_input)
        
        task_type = self.classify_input(user_input)
        
        if task_type == TaskType.REGISTER_CONTENT:
            self.registration_bot = RegistrationBot(self.conversation_history)
            response = {
                "text": "What type of learning material would you like to register?\n" + 
                       "\n".join(RegistrationBot.CONTENT_TYPES),
                "task_success": False
            }
            self.conversation_history.add_message("assistant", response["text"])
            return response
            
        # Handle other task types here...
        response = {
            "text": "I'll help you with that request.",
            "task_success": False
        }
        self.conversation_history.add_message("assistant", response["text"])
        return response

def register_content(content_text, content_type):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Generate a unique content_id based on current timestamp
    unique_content_id = int(time.time())

    # Extract first three words from content_text as title
    title = " ".join(content_text.split()[:3]) if content_text else "Untitled"

    # SQL Query to insert content
    query = """
    INSERT INTO content (content_id, class_id, title, type, description, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """
    
    # Execute query
    cursor.execute(query, (unique_content_id, 1, title, content_type, content_text))

    # Commit and close
    conn.commit()
    conn.close()

def insert_conversation(student_id, content_id, session_id, message):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create conversation object
    conversation_obj = {
        "session_id": session_id,
        "message": message,
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None  # Can be updated later
    }

    # Check if student-content pair exists
    cursor.execute(
        "SELECT conversations FROM conversation_list WHERE student_id = %s AND content_id = %s",
        (student_id, content_id)
    )
    result = cursor.fetchone()

    if result:
        # Append new conversation at the beginning of the JSON array
        conversations = json.loads(result[0])
        conversations.insert(0, conversation_obj)  # Insert latest at index 0
        cursor.execute(
            "UPDATE conversation_list SET conversations = %s WHERE student_id = %s AND content_id = %s",
            (json.dumps(conversations), student_id, content_id)
        )
    else:
        # Create a new entry if none exists
        cursor.execute(
            "INSERT INTO conversation_list (student_id, content_id, conversations) VALUES (%s, %s, %s)",
            (student_id, content_id, json.dumps([conversation_obj]))
        )

    conn.commit()
    conn.close()
    
def main():
    bot = EducationBot("F:\\S2_Framework\\AIM-HIGH\\BERT_Classification_Model")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            conversation = bot.conversation_history.get_conversation()
            insert_conversation(1,1,1,conversation)
            print("Conversation Stored Succesfully")
            break
            
        result = bot.process_input(user_input)
        print(f"Bot: {result['text']}")
        
        if result["task_success"]:
            print("Task completed successfully!")
            if "content" in result:
                register_content(result['content'],result['type'])
                print(f"Registered content: {result['content']}")
                print(f"Content Type: {result['type']}")

if __name__ == "__main__":
    main()