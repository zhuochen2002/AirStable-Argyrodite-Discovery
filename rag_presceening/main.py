"""
Solid-state battery RAG entrypoint: PDF processing, vector store, Q&A.
"""

import os
import json
import argparse
from typing import Optional
from pathlib import Path
import logging

from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from .rag_system import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SolidStateRAGSystem:
    """High-level solid-state RAG orchestration."""
    
    def __init__(self, 
                 papers_dir: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Args:
            papers_dir: PDF directory (default: package docs/papers)
            data_dir: Data output directory (default: package data)
            api_key: Intern S1 API key (optional; falls back to API/api_keys.json)
        """
        _BASE_DIR = Path(__file__).parent.resolve()
        if papers_dir is None:
            papers_dir = str(_BASE_DIR / "docs" / "papers")
        if data_dir is None:
            data_dir = str(_BASE_DIR / "data")
        self.papers_dir = Path(papers_dir)
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.pdf_processor = None
        self.vector_store = None
        self.rag_system = None
        
        logger.info("SolidStateRAGSystem initialized")
    
    def setup_components(self) -> None:
        """Wire PDF processor, vector store, and RAG."""
        logger.info("Setting up components...")
        
        self.pdf_processor = PDFProcessor(str(self.papers_dir))
        logger.info("PDFProcessor ready")
        
        self.vector_store = VectorStore(
            persist_dir=str(self.data_dir / "chroma_db")
        )
        logger.info("Vector store ready")
        
        self.rag_system = RAGSystem(
            vector_store=self.vector_store,
            runs_dir=self.data_dir / "rag_runs",
            api_key=self.api_key,
        )
        logger.info("RAG ready")
        logger.info("All components ready")
    
    def process_pdfs(self, force_reprocess: bool = False) -> bool:
        """
        Args:
            force_reprocess: Re-run PDF extraction even if JSON exists
            
        Returns:
            True on success
        """
        processed_file = self.data_dir / "processed_papers.json"
        
        if not force_reprocess and processed_file.exists():
            logger.info("processed_papers.json exists; skipping PDF step")
            return True
        
        try:
            logger.info("Processing PDFs...")
            documents = self.pdf_processor.process_all_pdfs()
            
            if documents:
                self.pdf_processor.save_processed_docs(str(processed_file))
                logger.info(f"Processed {len(documents)} PDFs")
                return True
            else:
                logger.error("No PDFs processed successfully")
                return False
                
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            return False
    
    def build_vector_database(self, force_rebuild: bool = False) -> bool:
        """
        Args:
            force_rebuild: Re-ingest even if collection is non-empty
            
        Returns:
            True on success
        """
        try:
            if not force_rebuild and self.vector_store.collection.count() > 0:
                logger.info("Vector DB already populated; skipping ingest")
                return True
            
            processed_file = self.data_dir / "processed_papers.json"
            if not processed_file.exists():
                logger.error("processed_papers.json missing; run process_pdfs first")
                return False
            
            with open(processed_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Ingesting {len(documents)} documents into vector store...")
            self.vector_store.ingest_processed_json(processed_json_path=str(processed_file))
            
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Vector DB build done: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Vector DB build failed: {str(e)}")
            return False
    
    def interactive_qa(self) -> None:
        """Interactive Q&A loop."""
        if not self.rag_system:
            logger.error("RAG not initialized; call setup_components first")
            return
        
        print("\n" + "="*60)
        print("Solid-state battery Q&A")
        print("="*60)
        print("Type 'quit' or 'exit' to leave")
        print("Type 'help' for help")
        print("Type 'stats' for collection stats")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nYour question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye.")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                
                elif not user_input:
                    continue
                
                print("\nThinking...")
                result = self.rag_system.ask(user_input)
                docs = result.get("retrieved_documents", [])
                
                print(f"\n📚 Retrieved {len(docs)} documents")
                print(f"❓ Q: {user_input}")
                print(f"💡 A: {result.get('expert_analysis', '')}")
                
                if docs:
                    print(f"\n📖 Top sources:")
                    for i, doc in enumerate(docs[:3]):
                        meta = doc.get("metadata", {})
                        print(f"  {i+1}. {meta.get('filename', 'N/A')}")
                        print(f"     keywords: {meta.get('keywords', 'N/A')}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                logger.error(f"Q&A error: {str(e)}")
                print(f"Sorry, something went wrong: {str(e)}")
    
    def _show_help(self) -> None:
        help_text = """
🤖 Help:

📚 Topics:
  • Solid-state batteries and electrolytes
  • Conductivity, interfaces, materials
  • Operating conditions and stability

🔍 Example questions:
  • "How can we improve ionic conductivity in solid electrolytes?"
  • "What solid electrolyte families are common?"
  • "What affects interface contact in solid-state cells?"
  • "What drives cycling stability?"

⚙️ Commands:
  • help: this message
  • stats: DB stats
  • quit/exit: leave

💡 More specific questions usually yield better answers.
        """
        print(help_text)
    
    def _show_system_stats(self) -> None:
        try:
            if self.vector_store:
                vdb_stats = self.vector_store.get_collection_stats()
                print(f"\n📊 Vector DB:")
                print(f"  chunks: {vdb_stats.get('count', vdb_stats.get('total_chunks', 'N/A'))}")
                print(f"  collection: {vdb_stats.get('collection', vdb_stats.get('collection_name', 'N/A'))}")
                print(f"  embedding model: {getattr(self.vector_store.model, 'name', 'all-MiniLM-L6-v2')}")
            
            runs_dir = self.data_dir / "rag_runs"
            if runs_dir.exists():
                try:
                    log_files = list(runs_dir.glob("*.jsonl"))
                    total_qa = 0
                    for f in log_files:
                        with open(f, encoding="utf-8") as fp:
                            total_qa += sum(1 for _ in fp)
                    print(f"\n💬 Run logs: {total_qa} lines")
                except Exception:
                    print(f"\n💬 Run logs: could not read")
                
        except Exception as e:
            logger.error(f"Stats error: {str(e)}")
            print("Could not read stats")
    
    def run_full_pipeline(self, force_reprocess: bool = False) -> bool:
        """
        Args:
            force_reprocess: Force PDF reprocess and rebuild
            
        Returns:
            True if pipeline succeeds
        """
        try:
            logger.info("Starting full pipeline...")
            
            self.setup_components()
            
            if not self.process_pdfs(force_reprocess):
                return False
            
            if not self.build_vector_database(force_reprocess):
                return False
            
            logger.info("Full pipeline finished")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False


def main():
    _BASE_DIR = Path(__file__).parent.resolve()
    
    parser = argparse.ArgumentParser(description="Solid-state battery RAG")
    parser.add_argument("--mode", choices=["full", "qa", "build"], default="full",
                       help="full: pipeline + QA; qa: QA only; build: pipeline only")
    parser.add_argument("--papers-dir", default=None,
                       help="PDF directory (default: package docs/papers)")
    parser.add_argument("--data-dir", default=None,
                       help="Data directory (default: package data)")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocess / rebuild")
    parser.add_argument("--api-key", help="Intern S1 API key (default: API/api_keys.json)")

    args = parser.parse_args()

    rag_system = SolidStateRAGSystem(
        papers_dir=args.papers_dir,
        data_dir=args.data_dir,
        api_key=args.api_key,
    )
    
    try:
        if args.mode == "full":
            if rag_system.run_full_pipeline(args.force):
                print("Build done. Starting interactive Q&A...")
                rag_system.interactive_qa()
            else:
                print("Build failed; see logs")
                
        elif args.mode == "qa":
            rag_system.setup_components()
            rag_system.interactive_qa()
            
        elif args.mode == "build":
            if rag_system.run_full_pipeline(args.force):
                print("Build done.")
            else:
                print("Build failed; see logs")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Run error: {str(e)}")
        print(f"Run error: {str(e)}")


if __name__ == "__main__":
    main()
