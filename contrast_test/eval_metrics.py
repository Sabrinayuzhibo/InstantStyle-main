import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import torch
from PIL import Image
from torchvision import transforms
from cleanfid import fid


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(img_dir: Path) -> List[Path]:
    return sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )


def _match_by_name(gen_dir: Path, ref_dir: Path) -> List[Tuple[Path, Path]]:
    gen_map: Dict[str, Path] = {p.name: p for p in _list_images(gen_dir)}
    ref_map: Dict[str, Path] = {p.name: p for p in _list_images(ref_dir)}
    common = sorted(set(gen_map.keys()) & set(ref_map.keys()))
    return [(gen_map[name], ref_map[name]) for name in common]


def _load_tensor(img_path: Path, device: torch.device, size: int = 256) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0).to(device)
    # lpips expects input in [-1, 1]
    return x * 2.0 - 1.0


def compute_lpips_mean(
    gen_dir: Path,
    lpips_ref_dir: Path,
    net: str,
    device: torch.device,
) -> Tuple[float, int]:
    pairs = _match_by_name(gen_dir, lpips_ref_dir)
    if not pairs:
        raise ValueError(
            f"No same-name image pairs found between {gen_dir} and {lpips_ref_dir}."
        )

    model = lpips.LPIPS(net=net).to(device)
    model.eval()

    vals: List[float] = []
    with torch.no_grad():
        for gen_path, ref_path in pairs:
            x = _load_tensor(gen_path, device)
            y = _load_tensor(ref_path, device)
            score = model(x, y).item()
            vals.append(score)

    return float(sum(vals) / len(vals)), len(vals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FID, art-FID and LPIPS for generated images."
    )
    parser.add_argument("--gen_dir", required=True, help="Generated image directory")
    parser.add_argument("--content_ref_dir", required=True, help="Reference content image directory for FID")
    parser.add_argument("--style_ref_dir", required=True, help="Reference style image directory for art-FID")
    parser.add_argument(
        "--lpips_ref_dir",
        default=None,
        help="Reference directory for LPIPS pairing (default: style_ref_dir)",
    )
    parser.add_argument(
        "--lpips_net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for clean-fid")
    parser.add_argument("--num_workers", type=int, default=0, help="Worker count for clean-fid")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device")

    args = parser.parse_args()

    gen_dir = Path(args.gen_dir).expanduser().resolve()
    content_ref_dir = Path(args.content_ref_dir).expanduser().resolve()
    style_ref_dir = Path(args.style_ref_dir).expanduser().resolve()
    lpips_ref_dir = (
        Path(args.lpips_ref_dir).expanduser().resolve() if args.lpips_ref_dir else style_ref_dir
    )

    for d in [gen_dir, content_ref_dir, style_ref_dir, lpips_ref_dir]:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, fallback to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # FID against content-domain real set
    fid_score = fid.compute_fid(
        str(gen_dir),
        str(content_ref_dir),
        mode="clean",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # art-FID against style-domain real set
    art_fid_score = fid.compute_fid(
        str(gen_dir),
        str(style_ref_dir),
        mode="clean",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    lpips_mean, lpips_pairs = compute_lpips_mean(
        gen_dir=gen_dir,
        lpips_ref_dir=lpips_ref_dir,
        net=args.lpips_net,
        device=device,
    )

    result = {
        "gen_dir": str(gen_dir),
        "content_ref_dir": str(content_ref_dir),
        "style_ref_dir": str(style_ref_dir),
        "lpips_ref_dir": str(lpips_ref_dir),
        "fid": float(fid_score),
        "art_fid": float(art_fid_score),
        "lpips": float(lpips_mean),
        "lpips_pairs": int(lpips_pairs),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
