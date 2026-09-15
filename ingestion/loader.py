import logging
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def load_documents(urls: list[str]) -> list[Document]:
    """Load web documents from given URLs using WebBaseLoader and SoupStrainer.
    
    Filters out nav, footer, script, and style tags to retain main content divs.
    Falls back to full page content if targeted strainer yields empty text.
    """
    documents: list[Document] = []
    
    # Strainer targeting main content containers
    strainer = SoupStrainer(name=["main", "article", "section", "div"])

    for url in urls:
        logger.info(f"Loading document from URL: {url}")
        loaded_docs: list[Document] = []
        try:
            loader = WebBaseLoader(
                web_path=url,
                header_template={"User-Agent": USER_AGENT},
                bs_kwargs={"parse_only": strainer},
            )
            loaded_docs = loader.load()
        except Exception as e:
            logger.warning(f"Failed to load {url} with SoupStrainer: {e}. Retrying without strainer...")

        # Fallback if empty or failed
        if not loaded_docs or not any(doc.page_content.strip() for doc in loaded_docs):
            logger.info(f"Fallback: loading full page for {url}")
            try:
                fallback_loader = WebBaseLoader(
                    web_path=url,
                    header_template={"User-Agent": USER_AGENT},
                )
                loaded_docs = fallback_loader.load()
            except Exception as e:
                logger.error(f"Failed to load {url}: {e}")
                continue

        for doc in loaded_docs:
            doc.metadata["source_url"] = url
            if "title" not in doc.metadata or not doc.metadata["title"]:
                doc.metadata["title"] = url.split("/")[-1].replace("-", " ").capitalize()
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} document(s) from {len(urls)} URLs")
    return documents
