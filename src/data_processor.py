# data_processor.py
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class MedicalDocument:
    """Cấu trúc dữ liệu y khoa đã được xử lý"""
    id: str
    title: str
    url: str
    content: str
    authors: List[str]
    reviewers: List[str]
    references: List[str]
    sections: Dict[str, str]
    metadata: Dict[str, Any]


class MedicalDataProcessor:
    """Xử lý dữ liệu y khoa từ MSD Manuals để training AI"""

    def __init__(self, data_dir: str = "data_msd"):
        self.data_dir = Path(data_dir)
        # Typo correction map
        self.typo_map = {
            "tiểu dường": "tiểu đường",
            "cao huyết ap": "cao huyết áp",
            "viem phoi": "viêm phổi",
            "viem phổi": "viêm phổi",
            "dau dau": "đau đầu",
            "dau đầu": "đau đầu",
        }

    def preprocess_query(self, query: str) -> str:
        """Tiền xử lý query trước khi search"""
        if not query or not isinstance(query, str):
            return ""
        
        # Giới hạn độ dài
        if len(query) > 500:
            query = query[:500]
        
        # Lowercase và strip
        query = query.lower().strip()
        
        # Fix typos
        for typo, correct in self.typo_map.items():
            query = query.replace(typo, correct)
        
        # Loại bỏ ký tự đặc biệt thừa
        query = re.sub(r'[^\w\s\u00C0-\u1EF9?]', ' ', query)
        
        # Loại bỏ nhiều space
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

    def filter_results_by_score(self, results: List[Dict], min_score: float = 0.3) -> List[Dict]:
        """Filter kết quả có score thấp"""
        filtered = []
        for result in results:
            score = result.get('rerank_score', 0) or result.get('similarity_score', 0)
            if score >= min_score:
                filtered.append(result)
        return filtered

    def clean_text(self, text: str) -> str:
        """Làm sạch text, loại bỏ ký tự thừa và lỗi encoding"""
        if not text:
            return ""

        # Fix các lỗi encoding phổ biến
        text = text.replace("hành v i", "hành vi")
        text = text.replace("v i ", "vi ")
        text = text.replace("ớn lạnh", "run lạnh")
        text = text.replace("chọc dọ", "chọc dò")
        text = text.replace("thài ra", "thải ra")

        # Loại bỏ các ký tự đặc biệt không cần thiết
        text = text.replace('|', ' ')

        # Loại bỏ nhiều space liên tiếp
        text = re.sub(r'\s+', ' ', text)

        # Loại bỏ space trước dấu câu
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        return text.strip()

    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Trích xuất metadata từ nội dung"""
        lines = content.split('\n')

        title = ""
        url = ""
        for line in lines[:5]:
            if line.startswith("TIÊU ĐỀ:"):
                title = line.replace("TIÊU ĐỀ:", "").strip()
            elif line.startswith("URL:"):
                url = line.replace("URL:", "").strip()

        return {
            'title': title,
            'url': url,
            'source': 'MSD Manuals',
            'language': 'vi'
        }

    def extract_authors_improved(self, content: str) -> tuple[List[str], List[str]]:
        """Trích xuất tác giả và người xem xét"""
        authors = []
        reviewers = []

        # Tìm phần tác giả
        author_pattern = r'Theo\s+(.*?)(?=Xem xét bởi|Đã xem xét)'
        author_match = re.search(author_pattern, content, re.DOTALL)

        if author_match:
            author_text = author_match.group(1)
            lines = [line.strip() for line in author_text.split('\n') if line.strip()]

            for line in lines:
                if line in [',', 'MD', 'PhD', 'MPH', 'MACP']:
                    continue
                if any(x in line for x in ['University', 'College', 'School', 'Hospital', 'Center']):
                    continue
                if 5 < len(line) < 50 and any(c.isalpha() for c in line):
                    name = line.rstrip(',').strip()
                    authors.append(name)
                    break

        # Tìm người xem xét
        reviewer_pattern = r'Xem xét bởi\s+(.*?)(?=Đã xem xét)'
        reviewer_match = re.search(reviewer_pattern, content, re.DOTALL)

        if reviewer_match:
            reviewer_text = reviewer_match.group(1)
            lines = [line.strip() for line in reviewer_text.split('\n') if line.strip()]

            for line in lines:
                if line in [',', 'MD', 'PhD', 'MPH', 'MACP']:
                    continue
                if any(x in line for x in ['University', 'College', 'School', 'Hospital', 'Center', 'Division']):
                    continue
                if 5 < len(line) < 50 and any(c.isalpha() for c in line):
                    name = line.rstrip(',').strip()
                    reviewers.append(name)
                    break

        return authors, reviewers

    def extract_main_content(self, content: str) -> str:
        """Trích xuất nội dung chính, loại bỏ header và footer"""
        main_start = content.find("NỘI DUNG:")
        if main_start == -1:
            return content

        main_content = content[main_start + len("NỘI DUNG:"):]

        # Tìm vị trí kết thúc metadata
        metadata_end_pattern = r'Đã xem xét/Đã chỉnh sửa.*?v\d+_vi'
        metadata_match = re.search(metadata_end_pattern, main_content, re.DOTALL)

        if metadata_match:
            main_content = main_content[metadata_match.end():]
        else:
            lines = main_content.split('\n')
            content_lines = []
            skip_metadata = True

            for line in lines:
                line_stripped = line.strip()

                if not line_stripped:
                    continue

                is_metadata = any(x in line_stripped for x in [
                    'Theo', 'Xem xét bởi', 'Đã xem xét', 'Đã chỉnh sửa',
                    'MD,', 'PhD,', 'University', 'College', 'Hospital',
                    'v_vi', 'Thg', 'đã sửa đổi'
                ])

                if not is_metadata:
                    skip_metadata = False

                if not skip_metadata:
                    content_lines.append(line_stripped)

            main_content = ' '.join(content_lines)

        # Loại bỏ footer
        footer_patterns = [
            "Test your Knowledge",
            "Take a Quiz!",
            "Bản quyền",
            "© 2025",
            "© 2024",
            "© 2023",
            "Merck & Co."
        ]

        for pattern in footer_patterns:
            idx = main_content.find(pattern)
            if idx != -1:
                main_content = main_content[:idx]
                break

        main_content = self.clean_text(main_content)
        return main_content

    def extract_references(self, content: str) -> List[str]:
        """Trích xuất các tham chiếu"""
        references = []

        pattern1 = r'\(Xem thêm\s+([^)]+)\)'
        matches = re.findall(pattern1, content)
        for match in matches:
            if 'và xem' in match or 'và' in match:
                parts = re.split(r'\s+và\s+(?:xem\s+)?', match)
                for part in parts:
                    part = part.strip('. ')
                    if 20 < len(part) < 150 and not any(x in part for x in ['Đã xem xét', 'đã sửa đổi', 'Thg', '_vi']):
                        references.append(self.clean_text(part))
            else:
                match = match.strip('. ')
                if 20 < len(match) < 150 and not any(x in match for x in ['Đã xem xét', 'đã sửa đổi', 'Thg', '_vi']):
                    references.append(self.clean_text(match))

        references = list(dict.fromkeys(references))
        return references[:10]

    def extract_sections(self, content: str) -> Dict[str, str]:
        """Trích xuất các phần của tài liệu"""
        sections = {}

        section_headers = [
            "Căn nguyên",
            "Sinh lý bệnh",
            "Triệu chứng và Dấu hiệu",
            "Triệu chứng",
            "Chẩn đoán",
            "Điều trị",
            "Phòng ngừa",
            "Những điểm chính",
            "Tiên lượng",
            "Phân loại",
            "Các biến chứng"
        ]

        header_positions = []
        for header in section_headers:
            pattern1 = rf'\|\s*{re.escape(header)}\s*\|'
            for match in re.finditer(pattern1, content):
                header_positions.append((match.end(), header))

            pattern2 = rf'{re.escape(header)}\s+([A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ])'
            for match in re.finditer(pattern2, content):
                text_after = content[match.start(1):match.start(1) + 50]
                is_another_header = any(h in text_after for h in section_headers if h != header)
                if not is_another_header:
                    header_positions.append((match.start(1), header))

        header_positions.sort()

        for i, (start_pos, header) in enumerate(header_positions):
            if i < len(header_positions) - 1:
                content_end = header_positions[i + 1][0]
            else:
                content_end = len(content)

            section_content = content[start_pos:content_end].strip()
            section_content = section_content.lstrip('| \n\t')

            if section_content and 50 < len(section_content) < 20000:
                sections[header] = self.clean_text(section_content)

        return sections

    def process_file(self, filepath: Path) -> MedicalDocument:
        """Xử lý một file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = self.extract_metadata(content)
        authors, reviewers = self.extract_authors_improved(content)
        main_content = self.extract_main_content(content)
        references = self.extract_references(content)
        sections = self.extract_sections(main_content)

        doc_id = filepath.stem.lower().replace(' ', '_').replace(',', '')

        return MedicalDocument(
            id=doc_id,
            title=metadata['title'],
            url=metadata['url'],
            content=main_content,
            authors=authors,
            reviewers=reviewers,
            references=references,
            sections=sections,
            metadata=metadata
        )

    def process_all(self) -> List[MedicalDocument]:
        """Xử lý tất cả file trong thư mục"""
        documents = []

        txt_files = sorted(self.data_dir.glob("*.txt"))
        print(f"Tìm thấy {len(txt_files)} file")

        for i, filepath in enumerate(txt_files, 1):
            try:
                doc = self.process_file(filepath)
                documents.append(doc)
                print(f"[{i}/{len(txt_files)}] ✓ {filepath.name}")
            except Exception as e:
                print(f"[{i}/{len(txt_files)}] ✗ {filepath.name}: {e}")

        return documents

    def save_to_json(self, documents: List[MedicalDocument], output_file: str = "processed_medical_data.json"):
        """Lưu dữ liệu đã xử lý ra JSON"""
        data = [asdict(doc) for doc in documents]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Đã lưu {len(documents)} documents vào {output_file}")
        return output_file

    def save_to_jsonl(self, documents: List[MedicalDocument], output_file: str = "processed_medical_data.jsonl"):
        """Lưu ra JSONL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                json.dump(asdict(doc), f, ensure_ascii=False)
                f.write('\n')

        print(f"✓ Đã lưu {len(documents)} documents vào {output_file}")
        return output_file

    def create_training_dataset(self, documents: List[MedicalDocument], output_file: str = "training_data.jsonl"):
        """Tạo dataset training"""
        training_data = []

        for doc in documents:
            training_data.append({
                "instruction": f"{doc.title} là gì?",
                "input": "",
                "output": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "metadata": {
                    "source": doc.url,
                    "title": doc.title
                }
            })

            for section_name, section_content in doc.sections.items():
                if len(section_content) > 50:
                    training_data.append({
                        "instruction": f"Giải thích về {section_name.lower()} của {doc.title}",
                        "input": "",
                        "output": section_content,
                        "metadata": {
                            "source": doc.url,
                            "title": doc.title,
                            "section": section_name
                        }
                    })

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        print(f"✓ Đã tạo {len(training_data)} training examples vào {output_file}")
        return output_file

    def generate_statistics(self, documents: List[MedicalDocument]):
        """Tạo thống kê"""
        total_chars = sum(len(doc.content) for doc in documents)
        total_sections = sum(len(doc.sections) for doc in documents)
        total_refs = sum(len(doc.references) for doc in documents)

        print("\n" + "=" * 60)
        print("THỐNG KÊ DỮ LIỆU:")
        print("=" * 60)
        print(f"📄 Tổng số documents: {len(documents)}")
        print(f"📝 Tổng số ký tự: {total_chars:,}")
        print(f"📊 Trung bình ký tự/doc: {total_chars // len(documents) if documents else 0:,}")
        print(f"🔖 Tổng số sections: {total_sections}")
        print(f"🔗 Tổng số references: {total_refs}")
        print(f"👥 Documents có tác giả: {sum(1 for doc in documents if doc.authors)}")
        print(f"✅ Documents có reviewer: {sum(1 for doc in documents if doc.reviewers)}")
        print("=" * 60)


def main():
    """Main function"""
    processor = MedicalDataProcessor("../data_msd")

    print("🚀 Bắt đầu xử lý dữ liệu y khoa...")
    print("=" * 60)

    documents = processor.process_all()

    if not documents:
        print("❌ Không tìm thấy document nào!")
        return

    print("\n💾 Đang lưu dữ liệu...")

    processor.save_to_json(documents, "medical_data.json")
    processor.save_to_jsonl(documents, "medical_data.jsonl")
    processor.create_training_dataset(documents, "training_data.jsonl")

    processor.generate_statistics(documents)

    if documents:
        print("\n" + "=" * 60)
        print("VÍ DỤ DOCUMENT ĐẦU TIÊN:")
        print("=" * 60)
        doc = documents[0]
        print(f"📌 ID: {doc.id}")
        print(f"📄 Title: {doc.title}")
        print(f"👤 Authors: {', '.join(doc.authors) if doc.authors else 'N/A'}")
        print(f"👥 Reviewers: {', '.join(doc.reviewers) if doc.reviewers else 'N/A'}")
        print(f"🔗 References: {len(doc.references)}")
        print(f"📑 Sections: {', '.join(doc.sections.keys())}")
        print(f"📝 Content preview: {doc.content[:200]}...")
        print("=" * 60)


if __name__ == "__main__":
    main()