from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["pdf_to_markdown", "convert_many", "PdfToMarkdownError"]


class PdfToMarkdownError(RuntimeError):
    """Lỗi phát sinh trong quá trình chuyển đổi."""


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def parse_pages(spec: str | None) -> list[int] | None:
    """'1-3,7,10-12' -> [0,1,2,6,9,10,11] (0-based, dùng cho pymupdf4llm)."""
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if start > end:
                raise ValueError(f"Khoảng trang không hợp lệ: {part}")
            pages.extend(range(start - 1, end))
        else:
            pages.append(int(part) - 1)
    return sorted(set(p for p in pages if p >= 0))


def _read_metadata(pdf_path: Path) -> dict:
    try:
        import pymupdf  # PyMuPDF >= 1.24 expose tên `pymupdf`
    except ImportError:  # pragma: no cover
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            return {}
    junk = {"", "untitled", "(anonymous)", "unknown", "none"}
    try:
        with pymupdf.open(pdf_path) as doc:
            meta = doc.metadata or {}
            clean = lambda k: (meta.get(k) or "").strip()
            return {
                "title": "" if clean("title").lower() in junk else clean("title"),
                "author": "" if clean("author").lower() in junk else clean("author"),
                "pages": doc.page_count,
            }
    except Exception:
        return {}


def _front_matter(pdf_path: Path, meta: dict) -> str:
    def esc(v: str) -> str:
        return '"' + str(v).replace('"', '\\"') + '"'

    lines = ["---", f"source: {esc(pdf_path.name)}"]
    if meta.get("title"):
        lines.append(f"title: {esc(meta['title'])}")
    if meta.get("author"):
        lines.append(f"author: {esc(meta['author'])}")
    if meta.get("pages"):
        lines.append(f"pages: {meta['pages']}")
    lines += ["---", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Hậu xử lý: bỏ header/footer lặp, nối dòng, dọn khoảng trắng
# --------------------------------------------------------------------------- #
_NUMERIC = re.compile(r"\d+")
_MD_SPECIAL = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||```|!\[|\$\$|---|===)")


def _normalize_line(line: str) -> str:
    """Chuẩn hoá để so khớp header/footer (số trang thay đổi từng trang)."""
    return _NUMERIC.sub("#", line.strip().lower())


def strip_running_heads(pages: Sequence[str], threshold: float = 0.6) -> list[str]:
    """Xoá header/footer lặp lại trên phần lớn các trang (tên tạp chí, số trang...)."""
    if len(pages) < 4:
        return list(pages)

    counter: Counter[str] = Counter()
    for text in pages:
        lines = [l for l in text.splitlines() if l.strip()]
        candidates = lines[:2] + lines[-2:]
        for line in set(_normalize_line(l) for l in candidates):
            if 0 < len(line) <= 120:
                counter[line] += 1

    repeated = {k for k, v in counter.items() if v >= max(3, threshold * len(pages))}
    if not repeated:
        return list(pages)

    cleaned = []
    for text in pages:
        lines = text.splitlines()
        head, tail = 0, len(lines)
        while head < tail and (not lines[head].strip() or _normalize_line(lines[head]) in repeated):
            head += 1
        while tail > head and (not lines[tail - 1].strip() or _normalize_line(lines[tail - 1]) in repeated):
            tail -= 1
        cleaned.append("\n".join(lines[head:tail]))
    return cleaned


def reflow_paragraphs(md: str) -> str:
    """Nối các dòng bị ngắt giữa đoạn (PDF xuống dòng theo cột) + gộp từ bị gạch nối."""
    out_blocks: list[str] = []
    in_code = False

    for block in md.split("\n\n"):
        lines = block.split("\n")
        if any(l.lstrip().startswith("```") for l in lines):
            in_code = not in_code
            out_blocks.append(block)
            continue
        if in_code or any(l.lstrip().startswith("|") for l in lines):
            out_blocks.append(block)
            continue

        merged: list[str] = []
        for line in lines:
            if not merged or _MD_SPECIAL.match(line) or not merged[-1].strip():
                merged.append(line)
                continue
            prev = merged[-1]
            if prev.rstrip().endswith("-") and not prev.rstrip().endswith("--"):
                merged[-1] = prev.rstrip()[:-1] + line.lstrip()   # infor- mation -> information
            else:
                merged[-1] = prev.rstrip() + " " + line.lstrip()
        out_blocks.append("\n".join(merged))

    return "\n\n".join(out_blocks)


def tidy(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\u00a0", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)          # bỏ khoảng trắng cuối dòng
    md = re.sub(r"\n{3,}", "\n\n", md)          # tối đa 1 dòng trống
    md = re.sub(r"(?m)^-{4,}$", "---", md)
    return md.strip() + "\n"


# Trích xuất bằng pymupdf4llm
def _extract(pdf_path: Path, pages: list[int] | None, image_dir: Path | None, dpi: int) -> str:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise PdfToMarkdownError("Thiếu thư viện: pip install pymupdf4llm") from exc

    kwargs = dict(
        page_chunks=True,                # tách theo trang để lọc header/footer
        table_strategy="lines_strict",   # nhận diện bảng theo đường kẻ, ít false positive
        ignore_images=image_dir is None,
    )
    if pages is not None:
        kwargs["pages"] = pages
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        kwargs.update(
            write_images=True,
            image_path=str(image_dir),
            image_format="png",
            dpi=dpi,
        )

    chunks = pymupdf4llm.to_markdown(str(pdf_path), **kwargs)
    page_texts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    page_texts = strip_running_heads(page_texts)
    return "\n\n".join(t.strip() for t in page_texts if t.strip())


# --------------------------------------------------------------------------- #
# API chính
# --------------------------------------------------------------------------- #
def pdf_to_markdown(
    pdf_path: str | Path,
    out_path: str | Path | None = None,
    *,
    pages: str | Sequence[int] | None = None,
    extract_images: bool = True,
    dpi: int = 200,
    reflow: bool = False,
    front_matter: bool = True,
) -> Path:
    """Chuyển 1 file PDF sang Markdown, trả về đường dẫn file .md đã ghi.

    Args:
        pdf_path:       đường dẫn file PDF đầu vào.
        out_path:       file .md hoặc thư mục đích. Mặc định: cùng chỗ với PDF.
        pages:          '1-5,8' hoặc list index 0-based. None = toàn bộ.
        extract_images: tách hình/biểu đồ ra thư mục <tên>_images/ và chèn link.
        dpi:            độ phân giải ảnh xuất ra.
        reflow:         nối các dòng bị ngắt trong đoạn văn (hợp cho paper 2 cột).
        front_matter:   thêm YAML metadata ở đầu file.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PdfToMarkdownError(f"Không phải file PDF: {pdf_path.name}")

    # Xác định file đích
    if out_path is None:
        md_path = pdf_path.with_suffix(".md")
    else:
        out_path = Path(out_path).expanduser()
        md_path = out_path if out_path.suffix.lower() == ".md" else out_path / f"{pdf_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    page_list = parse_pages(pages) if isinstance(pages, str) else (list(pages) if pages else None)
    image_dir = md_path.parent / f"{pdf_path.stem}_images" if extract_images else None

    md = _extract(pdf_path, page_list, image_dir, dpi)

    if reflow:
        md = reflow_paragraphs(md)
    md = tidy(md)

    if front_matter:
        md = _front_matter(pdf_path, _read_metadata(pdf_path)) + md

    md_path.write_text(md, encoding="utf-8")
    return md_path


def convert_many(paths: Iterable[Path], **kwargs) -> list[Path]:
    results: list[Path] = []
    for p in paths:
        try:
            out = pdf_to_markdown(p, **kwargs)
            print(f"[OK] {p.name} -> {out}")
            results.append(out)
        except Exception as exc:  # tiếp tục với các file còn lại
            print(f"[LỖI] {p.name}: {exc}", file=sys.stderr)
    return results


# CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chuyển PDF nghiên cứu sang Markdown (pymupdf4llm).")
    ap.add_argument("input", help="File .pdf hoặc thư mục chứa PDF")
    ap.add_argument("-o", "--output", help="File .md hoặc thư mục đích")
    ap.add_argument("-p", "--pages", help="Ví dụ: 1-5,8 (mặc định: tất cả)")
    ap.add_argument("--no-images", action="store_true", help="Không tách hình ảnh")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--reflow", action="store_true",
                    help="Nối dòng trong đoạn văn (khuyên dùng cho paper 2 cột)")
    ap.add_argument("--no-front-matter", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.input).expanduser()
    opts = dict(
        out_path=args.output,
        pages=args.pages,
        extract_images=not args.no_images,
        dpi=args.dpi,
        reflow=args.reflow,
        front_matter=not args.no_front_matter,
    )

    try:
        if src.is_dir():
            pdfs = sorted(src.rglob("*.pdf"))
            if not pdfs:
                print(f"Không có file PDF nào trong {src}", file=sys.stderr)
                return 1
            convert_many(pdfs, **opts)
        else:
            print(pdf_to_markdown(src, **opts))
    except (PdfToMarkdownError, FileNotFoundError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())