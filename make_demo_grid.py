#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


NAME_RE = re.compile(r"^\d+_sty_(?P<style>[^_]+)_cnt_(?P<content>.+?)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)

FIXED_CONTENT_OFFSET_X = 0
FIXED_CONTENT_OFFSET_Y = 300
FIXED_STYLE_OFFSET_X = 100
FIXED_STYLE_OFFSET_Y = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a demo grid: top row = style images, first column = content images, inner cells = generated results."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Directory containing generated images named like 0001_sty_0006_cnt_xxx.jpg",
    )
    parser.add_argument(
        "--style-dir",
        type=Path,
        default=Path("datasets/style-rank"),
        help="Directory containing style reference images like 0006.jpg",
    )
    parser.add_argument(
        "--content-dir",
        type=Path,
        default=Path("images_sd_xl_test_paper_cnt"),
        help="Directory containing content images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_grid.jpg"),
        help="Path to save output grid image",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default="",
        help="Optional comma-separated style ids. Example: 0006,0015,0016",
    )
    parser.add_argument(
        "--contents",
        type=str,
        default="",
        help="Optional comma-separated content ids (stems). Example: cxk,panda,Golden-Gate-Bridge",
    )
    parser.add_argument("--cell-size", type=int, default=256, help="Cell width/height in pixels")
    parser.add_argument("--gap", type=int, default=12, help="Gap between cells")
    parser.add_argument("--margin", type=int, default=24, help="Outer margin")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_rgba(text: str, font: ImageFont.ImageFont, fill: Tuple[int, int, int]) -> Image.Image:
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    left, top, right, bottom = probe_draw.textbbox((0, 0), text, font=font)
    w = max(1, right - left)
    h = max(1, bottom - top)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((-left, -top), text, font=font, fill=fill)
    return img


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def collect_pairs(result_dir: Path) -> Tuple[Dict[Tuple[str, str], Path], List[str], List[str]]:
    pair_map: Dict[Tuple[str, str], Path] = {}
    styles_order: List[str] = []
    contents_order: List[str] = []

    files = sorted([p for p in result_dir.iterdir() if p.is_file()])
    for p in files:
        m = NAME_RE.match(p.name)
        if not m:
            continue
        style_id = m.group("style")
        content_id = m.group("content")
        pair_map[(style_id, content_id)] = p
        if style_id not in styles_order:
            styles_order.append(style_id)
        if content_id not in contents_order:
            contents_order.append(content_id)

    if not pair_map:
        raise ValueError(f"No matched files found in {result_dir}")

    return pair_map, styles_order, contents_order


def parse_list_arg(value: str) -> List[str]:
    if not value.strip():
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def pick_path_with_ext(folder: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    args = parse_args()
    if not args.result_dir.exists():
        raise FileNotFoundError(f"result dir not found: {args.result_dir}")

    pair_map, auto_styles, auto_contents = collect_pairs(args.result_dir)

    styles = parse_list_arg(args.styles) or auto_styles
    contents = parse_list_arg(args.contents) or auto_contents

    cell = args.cell_size
    gap = args.gap
    margin = args.margin

    cols = 1 + len(styles)
    rows = 1 + len(contents)

    header_h = 70
    width = margin * 2 + cols * cell + (cols - 1) * gap
    height = margin * 2 + header_h + rows * cell + (rows - 1) * gap

    canvas = Image.new("RGBA", (width, height), (242, 242, 242, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(52)
    label_font = load_font(24)

    start_x = margin
    start_y = margin + header_h

    # Place "Content" and rotated "Style" around the same anchor for stable alignment.
    content_label = render_text_rgba("Content", label_font, (229, 147, 41))
    style_label = render_text_rgba("Style", title_font, (92, 140, 66)).rotate(90, expand=True)

    anchor_x = start_x + cell // 2
    anchor_y = margin + 6

    content_x = max(margin, anchor_x - style_label.width - content_label.width - 10)
    content_y = anchor_y + 2
    style_x = content_x + content_label.width + 10
    style_y = anchor_y

    content_x += FIXED_CONTENT_OFFSET_X
    content_y += FIXED_CONTENT_OFFSET_Y
    style_x += FIXED_STYLE_OFFSET_X
    style_y += FIXED_STYLE_OFFSET_Y

    canvas.alpha_composite(content_label, (content_x, content_y))
    canvas.alpha_composite(style_label, (style_x, style_y))

    # Draw style row.
    for c, style_id in enumerate(styles, start=1):
        x = start_x + c * (cell + gap)
        y = start_y
        style_path = pick_path_with_ext(args.style_dir, style_id)
        if style_path and style_path.exists():
            img = center_crop_resize(Image.open(style_path), cell)
            canvas.paste(img, (x, y))
        else:
            draw.rectangle((x, y, x + cell, y + cell), outline=(180, 180, 180), width=3)
            draw.text((x + 12, y + cell // 2 - 10), f"missing\n{style_id}", fill=(120, 120, 120), font=label_font)

    # Draw content column and generated cells.
    for r, content_id in enumerate(contents, start=1):
        y = start_y + r * (cell + gap)
        x0 = start_x
        content_path = pick_path_with_ext(args.content_dir, content_id)
        if content_path and content_path.exists():
            img = center_crop_resize(Image.open(content_path), cell)
            canvas.paste(img, (x0, y))
        else:
            draw.rectangle((x0, y, x0 + cell, y + cell), outline=(180, 180, 180), width=3)
            draw.text((x0 + 12, y + cell // 2 - 10), f"missing\n{content_id}", fill=(120, 120, 120), font=label_font)

        for c, style_id in enumerate(styles, start=1):
            x = start_x + c * (cell + gap)
            out_path = pair_map.get((style_id, content_id))
            if out_path and out_path.exists():
                img = center_crop_resize(Image.open(out_path), cell)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle((x, y, x + cell, y + cell), outline=(180, 180, 180), width=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(args.output)
    print(f"saved demo grid: {args.output}")
    print(f"styles={len(styles)} contents={len(contents)}")


if __name__ == "__main__":
    main()
