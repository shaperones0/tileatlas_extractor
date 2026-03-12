import math
from pathlib import Path
from collections.abc import Iterable

from PIL import Image
from pydantic import BaseModel
import rpack


def ceil_to(n: int, divisor: int) -> int:
    return int(math.ceil(n / divisor)) * divisor


def image_canvas_size(im: Image.Image, new_width: int, new_height: int) -> Image.Image:
    new_im = Image.new(im.mode, (new_width, new_height))
    new_im.paste(im, ((new_width - im.size[0]) // 2, (new_height - im.size[1]) // 2))
    return new_im


class SpritePacked(BaseModel):
    width: int
    height: int
    x: int = 0
    y: int = 0


def main(img_paths: Iterable[str], snap: int = 128) -> None:
    sprites: list[SpritePacked] = []
    ims: list[Image.Image] = []

    for img_path in img_paths:
        img_path = Path(img_path)
        img = Image.open(img_path)
        width, height = img.size
        width, height = ceil_to(width, snap), ceil_to(height, snap)

        # enlarge
        sprites.append(
            SpritePacked(
                width=width,
                height=height,
            )
        )
        ims.append(image_canvas_size(img, width, height))

    # pack and calc result
    sizes = ((sprite.width, sprite.height) for sprite in sprites)
    poses = rpack.pack(sizes)
    width, height = 0, 0
    for idx, pos in enumerate(poses):
        sprites[idx].x = pos[0]
        sprites[idx].y = pos[1]
        width = max(width, sprites[idx].width + sprites[idx].x)
        height = max(height, sprites[idx].height + sprites[idx].y)

    # write
    out_path = Path(__file__).parent / "output.png"
    out_im = Image.new("RGBA", (width, height))
    for idx, sprite in enumerate(sprites):
        out_im.paste(ims[idx], (sprite.x, sprite.y))
    out_im.save(out_path)


if __name__ == "__main__":

    # main(sys.argv[1:])
    main([str(path) for path in (Path(__file__).parent / "test2").iterdir()])
