#!/usr/bin/env python3
"""Test conversation history"""

import sys
sys.path.insert(0, 'src')

from generative_model import call_model, buffer_memory
from config import search_index

session_id = "test_user"

print("🧪 Test 1: First message (no history)")
print("="*50)
question1 = "Tên tôi là Quang"
docs, similarity = search_index(question1)
response1 = call_model(question1, docs, session_id, similarity)
print()

print("\n🧪 Test 2: Second message (should remember name)")
print("="*50)
question2 = "Tôi tên gì?"
docs, similarity = search_index(question2)
response2 = call_model(question2, docs, session_id, similarity)
print()

print("\n🧪 Test 3: Medical question with history")
print("="*50)
question3 = "Tôi bị đau răng"
docs, similarity = search_index(question3)
response3 = call_model(question3, docs, session_id, similarity)
print()

print("\n🧪 Test 4: Follow-up medical question")
print("="*50)
question4 = "Nên làm gì?"
docs, similarity = search_index(question4)
response4 = call_model(question4, docs, session_id, similarity)
print()

print("\n📊 Summary:")
print("="*50)
print(f"Total messages in buffer: {len(buffer_memory.chat_memory.messages)}")
print("\nExpected: 8 messages (4 questions + 4 answers)")

