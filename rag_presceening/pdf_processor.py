"""
PDF extraction and preprocessing for solid-state battery literature.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.resolve()
PAPERS_DIR = str(_BASE_DIR / "docs" / "papers")
PROCESSED_DOCS_DIR = str(_BASE_DIR / "data" / "processed_docs")

class PDFProcessor:
    """PDF text extraction pipeline."""

    def __init__(self, papers_dir: str = PAPERS_DIR):
        """
        Args:
            papers_dir: Directory containing PDF files
        """
        self.papers_dir = Path(papers_dir)
        self.processed_docs: List[Dict[str, Any]] = []

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract text with pdfplumber (preferred).

        Args:
            pdf_path: Path to PDF

        Returns:
            Dict with metadata and full_text, or None on failure
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content = []
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        text_content.append({
                            "page": page_num + 1,
                            "text": text.strip(),
                            "type": "text"
                        })
                    
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and any(any(cell for cell in row if cell) for row in table):
                            table_text = self._table_to_text(table)
                            if table_text.strip():
                                text_content.append({
                                    "page": page_num + 1,
                                    "text": table_text,
                                    "type": "table"
                                })

                raw_title = None
                try:
                    pm = getattr(pdf, "metadata", None) or {}
                    raw_title = (pm.get("Title") or pm.get("title") or "").strip()
                except Exception:
                    pass
                if not raw_title and text_content:
                    first_page_text = next(
                        (p.get("text", "") for p in text_content if p.get("page") == 1 and p.get("text")),
                        "",
                    )
                    if first_page_text:
                        first_line = first_page_text.split("\n")[0].strip()
                        if 15 <= len(first_line) <= 250 and "abstract" not in first_line.lower()[:20]:
                            raw_title = first_line
                original_title = (raw_title or "").strip() or pdf_path.stem

                metadata = {
                    "filename": pdf_path.name,
                    "title": pdf_path.stem,
                    "original_title": original_title,
                    "total_pages": len(pdf.pages),
                    "extraction_method": "pdfplumber",
                }

                full_text_parts = []
                for p in text_content:
                    page_no = p.get("page", 0)
                    page_txt = p.get("text", "")
                    if not page_txt:
                        continue
                    full_text_parts.append(f"[PAGE {page_no}]\n{page_txt}")

                return {
                    "metadata": metadata,
                    "content": text_content,
                    "full_text": "\n\n".join(full_text_parts),
                }

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
        
    def _table_to_text(self, table):
        """Convert table rows to plain text."""
        if not table:
            return ""
        
        filtered_table = []
        for row in table:
            filtered_row = [str(cell).strip() if cell else "" for cell in row]
            if any(filtered_row):
                filtered_table.append(filtered_row)
        
        if not filtered_table:
            return ""
        
        text_lines = []
        for row in filtered_table:
            row_text = " | ".join(cell for cell in row if cell)
            if row_text:
                text_lines.append(row_text)
        
        return "\n".join(text_lines)

    def extract_text_with_mupdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Fallback extraction with PyMuPDF.

        Args:
            pdf_path: Path to PDF

        Returns:
            Dict with metadata and full_text, or None on failure
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            text_content = []

            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text and text.strip():
                    text_content.append({
                        "page": page_num + 1,
                        "text": text.strip()
                    })

            raw_title = None
            try:
                m = doc.metadata or {}
                raw_title = (m.get("title") or m.get("Title") or "").strip()
            except Exception:
                pass
            doc.close()

            original_title = (raw_title or "").strip() or pdf_path.stem
            metadata = {
                "filename": pdf_path.name,
                "title": pdf_path.stem,
                "original_title": original_title,
                "total_pages": total_pages,
                "extraction_method": "pymupdf",
            }

            full_text_parts = []
            for p in text_content:
                page_no = p.get("page", 0)
                page_txt = p.get("text", "")
                if not page_txt:
                    continue
                full_text_parts.append(f"[PAGE {page_no}]\n{page_txt}")

            return {
                "metadata": metadata,
                "content": text_content,
                "full_text": "\n\n".join(full_text_parts),
            }

        except Exception as e:
            logger.error(f"PyMuPDF error on {pdf_path}: {str(e)}")
            return None

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace and strip noise while keeping CJK and scientific symbols.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = re.sub(r"(?<=\w)-\n(?=\s*\w)", "", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
        
        allowed_chars = (
            r"0-9A-Za-z\u4e00-\u9fff"
            r"\s\.\,\;\:\!\?\-\(\)\[\]\{\}"
            r"℃°±×∙→←≥≤%/·–—"
            r"αβγδεζηθικλμνξοπρστυφχψω"
            r"ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
            r"²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"
            r"≈≠≤≥±∞∑∏∫∂∇∆√∛∜"
        )

        text = re.sub(f"[^{allowed_chars}]", "", text)

        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def extract_keywords(self, text: str) -> List[str]:
        """
        Collect domain keywords present in text (English terms for solid-state battery literature).
        """
        keywords: List[str] = []
        solid_state_keywords = [
            "solid-state", "solid electrolyte", "LLZO", "NASICON", "LATP", "LAGP",
            "solid-state battery", "SSB", "lithium metal battery",
            "electrolyte", "ion conductivity", "ionic conductivity", "interface",
            "conductivity", "temperature", "performance", "efficiency", "capacity",
            "energy density", "power density", "voltage", "current density",
            "cycling stability", "cycle life", "thermal stability", "doping",
            "garnet", "perovskite", "interfacial", "dendrite", "SEI", "lithium plating",
        ]
        text_lower = text.lower()
        for kw in solid_state_keywords:
            if kw.lower() in text_lower:
                keywords.append(kw)
        return list(set(keywords))

    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Process every PDF under papers_dir.

        Returns:
            List of processed document dicts
        """
        if not self.papers_dir.exists():
            logger.error(f"Papers directory not found: {self.papers_dir}")
            return []

        pdf_files = list(self.papers_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        processed_docs: List[Dict[str, Any]] = []

        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            logger.info(f"Processing: {pdf_file.name}")

            doc_data = self.extract_text_from_pdf(pdf_file)

            if doc_data is None:
                doc_data = self.extract_text_with_mupdf(pdf_file)

            if doc_data:
                doc_data["full_text"] = self.clean_text(doc_data["full_text"])

                doc_data["keywords"] = self.extract_keywords(doc_data["full_text"])

                doc_data["text_length"] = len(doc_data["full_text"])

                processed_docs.append(doc_data)
                logger.info(f"OK: {pdf_file.name}, length: {doc_data['text_length']}")
            else:
                logger.warning(f"Failed: {pdf_file.name}")

        self.processed_docs = processed_docs
        logger.info(f"Processed {len(processed_docs)} PDFs total")

        return processed_docs

    def save_processed_docs(self, output_file: str = f"{PROCESSED_DOCS_DIR}/processed_papers.json") -> None:
        """
        Save processed documents to JSON.

        Args:
            output_file: Output path
        """
        import json

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_data: List[Dict[str, Any]] = []
        for doc in self.processed_docs:
            save_doc = {
                "metadata": doc["metadata"],
                "keywords": doc.get("keywords", []),
                "text_length": doc.get("text_length", 0),
                "full_text": doc.get("full_text", ""),
            }
            save_data.append(save_doc)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    processor = PDFProcessor()
    docs = processor.process_all_pdfs()
    processor.save_processed_docs()

    print(f"Done. Processed {len(docs)} PDFs")
    for doc in docs[:3]:
        print(f"file: {doc['metadata']['filename']}")
        print(f"keywords: {doc.get('keywords', [])}")
        print(f"length: {doc.get('text_length', 0)}")
        print("-" * 50)
