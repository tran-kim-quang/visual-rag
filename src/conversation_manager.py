# conversation_manager.py
from typing import List, Dict, Optional
from collections import defaultdict
import json

class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.history: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = max_history
    
    def add_to_history(self, session_id: str, question: str, answer: str):
        """Lưu Q&A vào history"""
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append({
            "role": "user",
            "content": question
        })
        self.history[session_id].append({
            "role": "assistant", 
            "content": answer
        })
        
        # Giới hạn history
        if len(self.history[session_id]) > self.max_history * 2:
            self.history[session_id] = self.history[session_id][-self.max_history * 2:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Lấy history của session"""
        return self.history.get(session_id, [])
    
    def get_context_string(self, session_id: str, last_n: int = 3) -> str:
        """Lấy context dạng string cho prompt"""
        messages = self.get_history(session_id)
        if not messages:
            return ""
        
        # Lấy n cặp Q&A cuối
        recent = messages[-(last_n * 2):]
        
        context_parts = []
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                q = recent[i]["content"]
                a = recent[i + 1]["content"]
                context_parts.append(f"Q: {q}\nA: {a}")
        
        return "\n\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """Xóa history của session"""
        if session_id in self.history:
            del self.history[session_id]
    
    def has_history(self, session_id: str) -> bool:
        """Check xem session có history không"""
        return session_id in self.history and len(self.history[session_id]) > 0