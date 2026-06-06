import io
import logging

import docx
import pypdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt"}


def parse_file(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式：{ext}，请上传 pdf / docx / txt")

    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext == "docx":
        return parse_docx(file_bytes)
    else:
        return file_bytes.decode("utf-8", errors="ignore")


def parse_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    texts = [page.extract_text().strip() for page in reader.pages if page.extract_text()]
    logger.info(f"PDF 解析完成，共 {len(reader.pages)} 页")
    return "\n".join(texts)


def parse_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    logger.info(f"Word 解析完成，共 {len(texts)} 段")
    return "\n".join(texts)