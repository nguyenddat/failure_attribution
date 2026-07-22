"""
Convert PDF -> Markdown, bỏ qua ảnh.

Cài đặt:
    pip install pymupdf4llm

Dùng:
    python pdf_to_md.py input.pdf                 # in ra file input.md cạnh file gốc
    python pdf_to_md.py input.pdf output.md        # chỉ định đường dẫn output
"""

import sys
from pathlib import Path

import pymupdf4llm


def pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """
    Convert 1 file PDF sang Markdown, không nhúng ảnh.

    Args:
        pdf_path: đường dẫn tới file .pdf
        output_path: đường dẫn file .md muốn ghi ra. Nếu None thì tự suy
                      ra bằng cách đổi phần mở rộng của pdf_path.

    Returns:
        Nội dung markdown (string).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

    # write_images=False -> bỏ qua toàn bộ ảnh, chỉ lấy text + cấu trúc (heading, bảng, list...)
    md_text = pymupdf4llm.to_markdown(str(pdf_path), write_images=False)

    out = Path(output_path) if output_path else pdf_path.with_suffix(".md")
    out.write_text(md_text, encoding="utf-8")

    return md_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python pdf_to_md.py <input.pdf> [output.md]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_md = sys.argv[2] if len(sys.argv) > 2 else None

    pdf_to_markdown(input_pdf, output_md)
    print(
        f"Đã convert xong: {input_pdf} -> {output_md or Path(input_pdf).with_suffix('.md')}"
    )
