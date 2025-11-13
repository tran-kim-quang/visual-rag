# query_rewriter.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Union
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

class QueryRewriter:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model_name = os.getenv("SUMMARIZE_MODEL") or "kwaipilot/kat-coder-pro:free"
        
        if not os.getenv("SUMMARIZE_MODEL"):
            print(f"[WARNING] SUMMARIZE_MODEL không tồn tại, dùng mặc định: {self.model_name}")
    
    def rewrite_query(self, question: str, history: List[Union[HumanMessage, AIMessage, Dict]]) -> str:
        """Viết lại câu hỏi dựa trên context"""
        # Validate input
        if not question or not question.strip():
            return question
            
        if not history or len(history) == 0:
            return question
        
        # Build history
        try:
            history_text = self._build_history_text(history)
            if not history_text or len(history_text) < 10:
                return question
        except Exception as e:
            print(f"[ERROR] Build history failed: {e}")
            return question
        
        prompt = f"""Lịch sử hội thoại:
{history_text}

Câu hỏi mới: {question}

Viết lại câu hỏi thành 1 câu độc lập, rõ ràng để tìm kiếm tài liệu y tế (tiếng Việt, ngắn gọn). CHỈ TRẢ VỀ CÂU HỎI, KHÔNG GIẢI THÍCH."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
                timeout=5
            )
            
            rewritten = completion.choices[0].message.content.strip()
            rewritten = self._clean_output(rewritten)
            
            # Validate
            if not rewritten or len(rewritten) < 5 or len(rewritten) > 300:
                print(f"[WARNING] Invalid rewritten query, fallback")
                return question
            
            return rewritten
            
        except Exception as e:
            print(f"[ERROR Query Rewrite] {e}, fallback to original")
            return question
    
    def _build_history_text(self, history: List[Union[HumanMessage, AIMessage, Dict]]) -> str:
        """Build history text - support both LangChain messages and dicts"""
        history_parts = []
        recent = history[-4:]
        
        for msg in recent:
            try:
                # LangChain message format
                if hasattr(msg, 'content'):
                    role = "User" if isinstance(msg, HumanMessage) else "Bot"
                    content = msg.content
                # Dict format
                elif isinstance(msg, dict):
                    role = "User" if msg.get("role") == "user" else "Bot"
                    content = msg.get("content", "")
                else:
                    continue
                
                # Truncate
                if len(content) > 150:
                    content = content[:150] + "..."
                
                history_parts.append(f"{role}: {content}")
            except Exception as e:
                print(f"[WARNING] Skip invalid message: {e}")
                continue
        
        return "\n".join(history_parts)
    
    def _clean_output(self, text: str) -> str:
        """Clean rewritten query"""
        # Remove quotes
        text = text.strip('"\'')
        
        # Remove prefixes
        prefixes = ["Câu hỏi:", "Query:", "Rewritten:", "Answer:", "Viết lại:"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        return text.strip()


# Global instance
_rewriter = None

def rewrite_query(question: str, history: list) -> str:
    """Legacy function for backward compatibility"""
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    
    try:
        return _rewriter.rewrite_query(question, history)
    except Exception as e:
        print(f"[ERROR] Query rewrite failed: {e}")
        return question