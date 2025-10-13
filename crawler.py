import requests
import os
import random
import time
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET


class MSDCrawler:
    def __init__(self, base_url='https://www.msdmanuals.com', data_dir='data'):
        self.base_url = base_url
        self.data_dir = data_dir
        self.visited_urls = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"✓ Đã tạo thư mục: '{data_dir}'")

    def save_progress(self):
        """Lưu tiến trình để có thể tiếp tục sau"""
        progress_file = os.path.join(self.data_dir, 'progress.json')
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.visited_urls), f, ensure_ascii=False, indent=2)

    def load_progress(self):
        """Tải tiến trình đã lưu"""
        progress_file = os.path.join(self.data_dir, 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                self.visited_urls = set(json.load(f))
            print(f"✓ Đã tải {len(self.visited_urls)} URL đã cào từ lần trước")

    def get_page(self, url, retry=3):
        """Lấy nội dung trang web với retry"""
        for attempt in range(retry):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == retry - 1:
                    print(f"  ✗ Lỗi sau {retry} lần thử: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def get_urls_from_sitemap(self, sitemap_url):
        """Lấy URLs từ sitemap XML (hỗ trợ sitemap index)"""
        print(f"\n→ Đang phân tích sitemap: {sitemap_url}")
        response = self.get_page(sitemap_url)
        if not response:
            return []

        try:
            root = ET.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = []

            # Kiểm tra xem đây có phải sitemap index không
            sitemaps = root.findall('ns:sitemap', namespace)
            if sitemaps:
                print(f"  ✓ Đây là sitemap index, tìm thấy {len(sitemaps)} sitemap con")
                # Đệ quy vào từng sitemap con
                for sitemap in sitemaps:
                    loc = sitemap.find('ns:loc', namespace)
                    if loc is not None:
                        sub_url = loc.text
                        # CHỈ lấy sitemap topic (bài viết y khoa)
                        if 'topic.xml' not in sub_url:
                            continue
                        print(f"  → Đang cào sitemap con: {sub_url.split('/')[-1]}")
                        sub_urls = self.get_urls_from_sitemap(sub_url)
                        urls.extend(sub_urls)
                return urls

            # Nếu không phải index, lấy URLs bình thường
            for url_element in root.findall('ns:url', namespace):
                loc = url_element.find('ns:loc', namespace)
                if loc is not None:
                    # Chỉ lấy URLs professional
                    if '/professional/' in loc.text:
                        urls.append(loc.text)

            print(f"  ✓ Tìm thấy {len(urls)} URLs professional")
            return urls
        except Exception as e:
            print(f"  ✗ Lỗi phân tích XML: {e}")
            return []

    def extract_article_content(self, soup, url):
        """Trích xuất nội dung bài viết (cải tiến)"""
        data = {
            'url': url,
            'title': '',
            'content': '',
            'sections': []
        }

        # Lấy tiêu đề
        title_tag = soup.find('h1')
        if title_tag:
            data['title'] = title_tag.get_text(strip=True)

        # Lấy nội dung chính - thử nhiều selector
        content_selectors = [
            ('div', {'class': 'topic__content'}),
            ('div', {'class': 'content-body'}),
            ('article', {}),
            ('main', {})
        ]

        for tag, attrs in content_selectors:
            content_div = soup.find(tag, attrs)
            if content_div:
                # Lấy toàn bộ text
                data['content'] = content_div.get_text(separator='\n\n', strip=True)

                # Tách thành các section nếu có headers
                sections = content_div.find_all(['h2', 'h3', 'h4'])
                for section in sections:
                    section_title = section.get_text(strip=True)
                    section_content = []

                    # Lấy nội dung sau header cho đến header tiếp theo
                    for sibling in section.find_next_siblings():
                        if sibling.name in ['h2', 'h3', 'h4']:
                            break
                        text = sibling.get_text(strip=True)
                        if text:
                            section_content.append(text)

                    if section_content:
                        data['sections'].append({
                            'title': section_title,
                            'content': '\n\n'.join(section_content)
                        })
                break

        return data

    def save_article(self, data):
        """Lưu bài viết với format cải tiến"""
        if not data['title'] or not data['content']:
            return False

        # Tạo tên file an toàn
        safe_title = "".join(c for c in data['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title[:100]  # Giới hạn độ dài
        filename = f"{safe_title}.txt"
        filepath = os.path.join(self.data_dir, filename)

        # Tránh ghi đè
        counter = 1
        while os.path.exists(filepath):
            filename = f"{safe_title}_{counter}.txt"
            filepath = os.path.join(self.data_dir, filename)
            counter += 1

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"TIÊU ĐỀ: {data['title']}\n")
                f.write(f"URL: {data['url']}\n")
                f.write("=" * 80 + "\n\n")

                # Ghi nội dung chính
                f.write("NỘI DUNG:\n\n")
                f.write(data['content'])
                f.write("\n\n")

                # Ghi các sections nếu có
                if data['sections']:
                    f.write("=" * 80 + "\n")
                    f.write("CÁC PHẦN CHI TIẾT:\n\n")
                    for i, section in enumerate(data['sections'], 1):
                        f.write(f"\n--- {i}. {section['title']} ---\n\n")
                        f.write(section['content'])
                        f.write("\n")

            return True
        except Exception as e:
            print(f"  ✗ Lỗi lưu file: {e}")
            return False

    def crawl_article(self, url):
        """Cào một bài viết"""
        if url in self.visited_urls:
            return False

        response = self.get_page(url)
        if not response:
            return False

        soup = BeautifulSoup(response.content, 'html.parser')
        data = self.extract_article_content(soup, url)

        if self.save_article(data):
            self.visited_urls.add(url)
            return True
        return False

    def crawl_from_sitemap(self, sitemap_url, limit=None):
        """Cào từ sitemap"""
        print("\n" + "=" * 80)
        print("BẮT ĐẦU CÀO TỪ SITEMAP")
        print("=" * 80)

        urls = self.get_urls_from_sitemap(sitemap_url)
        if not urls:
            print("✗ Không lấy được URLs từ sitemap")
            return

        # Giới hạn số lượng nếu có
        if limit:
            urls = urls[:limit]
            print(f"\n→ Giới hạn cào {limit} bài đầu tiên")

        success_count = 0
        total = len(urls)

        for i, url in enumerate(urls, 1):
            # Bỏ qua URLs đã cào
            if url in self.visited_urls:
                continue

            print(f"\n[{i}/{total}] Đang cào: {url.split('/')[-1][:60]}...")

            if self.crawl_article(url):
                success_count += 1
                print(f"  ✓ Thành công (Tổng: {success_count}/{total})")
            else:
                print(f"  ✗ Thất bại hoặc đã cào")

            # Lưu tiến trình mỗi 10 bài
            if i % 10 == 0:
                self.save_progress()

            # Delay để không quá tải server
            time.sleep(random.uniform(1, 2))

        self.save_progress()
        print(f"\n✓ Hoàn thành! Đã cào thành công {success_count}/{total} bài viết")

    def crawl_category_page(self, url):
        """Cào trang danh mục (category) để lấy links các bài viết con"""
        print(f"\n→ Đang phân tích trang danh mục: {url}")
        response = self.get_page(url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        article_links = []

        # Tìm tất cả links trong trang
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Chỉ lấy links bài viết (có đường dẫn hợp lệ)
            if '/vi/professional/' in href and href not in article_links:
                full_url = urljoin(self.base_url, href)
                article_links.append(full_url)

        print(f"  ✓ Tìm thấy {len(article_links)} links")
        return article_links


# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    print("=" * 80)
    print("MSD MANUALS CRAWLER - PHIÊN BẢN CÀI TIẾN")
    print("=" * 80)

    crawler = MSDCrawler(data_dir='data_msd')

    # Tải tiến trình cũ nếu có
    crawler.load_progress()

    # Sitemap chính cho tiếng Việt
    SITEMAP_URL = 'https://www.msdmanuals.com/vi/sitemap.xml'

    # Bắt đầu cào
    crawler.crawl_from_sitemap(SITEMAP_URL)

    print("\n" + "=" * 80)
    print("HOÀN THÀNH!")
    print(f"Tổng số bài viết đã cào: {len(crawler.visited_urls)}")
    print(f"Dữ liệu được lưu tại: '{crawler.data_dir}'")
    print("=" * 80)