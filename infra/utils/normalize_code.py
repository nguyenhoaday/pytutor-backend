def normalize_code(code: str) -> str:
    """Chuẩn hoá code để lưu/so sánh ổn định.

    Mục tiêu:
    - Bỏ khoảng trắng thừa ở cuối mỗi dòng (tránh diff/embedding nhiễu).
    - Đảm bảo kết thúc bằng newline để xử lý đồng nhất giữa các nguồn input.
    """

    if not code:
        return code

    lines = [line.rstrip() for line in code.splitlines()]
    normalized = "\n".join(lines)

    if normalized and not normalized.endswith("\n"):
        normalized += "\n"

    return normalized