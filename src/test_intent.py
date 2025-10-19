"""Script để test phân loại intent dựa trên similarity score"""
from generative_model import classify_by_similarity, SIMILARITY_THRESHOLD
from config import search_index

def test_with_real_retrieval():
    """Test với retrieval thật từ tài liệu y tế"""
    print("\n" + "="*60)
    print("TEST PHÂN LOẠI INTENT VỚI SIMILARITY SCORE")
    print(f"Ngưỡng (threshold): {SIMILARITY_THRESHOLD}")
    print("Chỉ có 2 intent: medical_question và chitchat")
    print("="*60 + "\n")
    
    test_cases = [
        # Chitchat - không liên quan y tế
        ("Xin chào", "chitchat"),
        ("Hôm nay thời tiết đẹp quá", "chitchat"),
        ("Bạn có thích ăn phở không?", "chitchat"),
        ("Cảm ơn bạn", "chitchat"),
        
        # Medical questions - liên quan y tế
        ("Triệu chứng của bệnh tiểu đường là gì?", "medical_question"),
        ("Làm thế nào để điều trị cảm cúm?", "medical_question"),
        ("Bệnh viêm gan B có nguy hiểm không?", "medical_question"),
        ("Tôi bị đau đầu thì nên làm gì?", "medical_question"),
    ]
    
    for question, expected_intent in test_cases:
        # Retrieval để lấy similarity score
        docs, similarity_score = search_index(question)
        
        # Phân loại
        predicted_intent = classify_by_similarity(similarity_score)
        is_correct = predicted_intent == expected_intent
        
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"\n{status} Câu hỏi: '{question}'")
        print(f"     Similarity: {similarity_score:.3f}")
        print(f"     Dự đoán: {predicted_intent} | Mong đợi: {expected_intent}")
        
        if not is_correct:
            print(f"     [Lưu ý: Có thể do ngưỡng {SIMILARITY_THRESHOLD} cần điều chỉnh]")

def test_threshold_sensitivity():
    """Test với các ngưỡng khác nhau"""
    print("\n\n" + "="*60)
    print("TEST ĐỘ NHẠY CỦA NGƯỠNG")
    print("="*60 + "\n")
    
    test_questions = [
        "Triệu chứng của bệnh tiểu đường?",
        "Tôi bị đau đầu",
        "Hôm nay thời tiết thế nào?"
    ]
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for question in test_questions:
        print(f"\nCâu hỏi: '{question}'")
        docs, similarity_score = search_index(question)
        print(f"Similarity score: {similarity_score:.3f}")
        print("Phân loại với các ngưỡng khác nhau:")
        
        for threshold in thresholds:
            if similarity_score >= threshold:
                intent = "medical_question"
            else:
                intent = "chitchat"
            print(f"  - Ngưỡng {threshold}: {intent}")

if __name__ == "__main__":
    test_with_real_retrieval()
    test_threshold_sensitivity()

