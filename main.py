from pathlib import Path
import itertools as it

import networkx as nx
import numpy as np
from PIL import Image
import orjson
from pydantic import BaseModel


class SpriteBox(BaseModel):
    idx: int
    points: tuple[tuple[int, int], ...]
    bbox_left: int
    bbox_top: int
    bbox_right: int
    bbox_bottom: int


def box_intersect(box1: SpriteBox, box2: SpriteBox, leeway: int = 2) -> bool:
    x1 = max(box1.bbox_left, box2.bbox_left) - leeway
    y1 = max(box1.bbox_top, box2.bbox_top) - leeway
    x2 = min(box1.bbox_right, box2.bbox_right) + leeway
    y2 = min(box1.bbox_bottom, box2.bbox_bottom) + leeway
    return x1 <= x2 and y1 <= y2


def box_union(*boxes: SpriteBox) -> SpriteBox:
    new_points: list[tuple[int, int]] = list(boxes[0].points)
    bbox_left = boxes[0].bbox_left
    bbox_top = boxes[0].bbox_top
    bbox_right = boxes[0].bbox_right
    bbox_bottom = boxes[0].bbox_bottom
    for box in boxes[1:]:
        new_points.extend(box.points)
        bbox_left = min(bbox_left, box.bbox_left)
        bbox_top = min(bbox_top, box.bbox_top)
        bbox_right = max(bbox_right, box.bbox_right)
        bbox_bottom = max(bbox_bottom, box.bbox_bottom)
    return SpriteBox(
        idx=boxes[0].idx,
        points=tuple(new_points),
        bbox_left=bbox_left,
        bbox_top=bbox_top,
        bbox_right=bbox_right,
        bbox_bottom=bbox_bottom,
    )


def main(img_path_str: str, threshold: int = 1):
    img_path = Path(img_path_str)
    out_dir = img_path.parent / f"output-{img_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path)
    height, width = img.size
    arr = np.array(img)

    json_path = out_dir / "sets.json"
    if not json_path.exists():
        alpha = arr[:, :, 3] > threshold
        graph = nx.Graph()
        print("Read", img_path.name)

        idx_opaque = np.nonzero(alpha)
        print("Total:", len(idx_opaque[0]))
        for i, (r, c) in enumerate(zip(*idx_opaque)):
            r, c = int(r), int(c)
            if i % 100000 == 0:
                print(i)
            # populate edges
            if r > 0 and alpha[r - 1][c]:
                graph.add_edge((r, c), (r - 1, c))
            if r < height - 1 and alpha[r + 1][c]:
                graph.add_edge((r, c), (r + 1, c))
            if c > 0 and alpha[r][c - 1]:
                graph.add_edge((r, c), (r, c - 1))
            if c < width - 1 and alpha[r][c + 1]:
                graph.add_edge((r, c), (r, c + 1))

        print("Graph populated:", len(graph))
        sets = [list(s) for s in nx.connected_components(graph)]
        print("Sets calculated:", len(sets))
        jsonb = orjson.dumps(sets)
        json_path.write_bytes(jsonb)
        print("JSON written")
    else:
        jsonb = json_path.read_bytes()
        sets = orjson.loads(jsonb)
        print("Sets loaded:", len(sets))

    # calculate boxes
    boxes: list[SpriteBox] = []
    for idx, comp in enumerate(sets):
        max_alpha = max(arr[point[0], point[1], 3] for point in comp)
        if max_alpha < 50:
            print(f"Skipping {idx} - too transparent")
            continue
        bbox_left = comp[0][0]
        bbox_top = comp[0][1]
        bbox_right = comp[0][0]
        bbox_bottom = comp[0][1]
        for point in comp:
            bbox_left = min(bbox_left, point[0])
            bbox_top = min(bbox_top, point[1])
            bbox_right = max(bbox_right, point[0])
            bbox_bottom = max(bbox_bottom, point[1])
        boxes.append(
            SpriteBox(
                idx=idx,
                bbox_left=bbox_left,
                bbox_top=bbox_top,
                bbox_right=bbox_right,
                bbox_bottom=bbox_bottom,
                points=tuple((int(p[0]), int(p[1])) for p in comp),
            )
        )

    # merge boxes
    i = 0
    while True:
        i += 1
        prev_len = len(boxes)

        graph = nx.Graph()
        idx2box: dict[int, SpriteBox] = {}
        for box in boxes:
            graph.add_node(box.idx)
            assert box.idx not in idx2box
            idx2box[box.idx] = box
        for box1, box2 in it.combinations(boxes, 2):
            if box_intersect(box1, box2):
                graph.add_edge(box1.idx, box2.idx)

        sets = [list(s) for s in nx.connected_components(graph)]
        new_boxes: list[SpriteBox] = []
        for box_set in sets:
            new_boxes.append(box_union(*(idx2box[idx] for idx in box_set)))
        boxes = new_boxes
        print(f"Merging iteration {i}: {prev_len}->{len(boxes)}")
        if prev_len == len(boxes):
            break

    # write images
    for box in boxes:
        fn = f"{box.idx} {box.bbox_left};{box.bbox_top} - {box.bbox_right};{box.bbox_bottom}.png"
        print("Component", fn)

        new_arr = np.zeros(
            shape=(
                box.bbox_right - box.bbox_left + 1,
                box.bbox_bottom - box.bbox_top + 1,
                4,
            ),
            dtype=np.uint8,
        )
        for point in box.points:
            new_arr[point[0] - box.bbox_left, point[1] - box.bbox_top] = arr[
                point[0], point[1]
            ]

        new_img_path = out_dir / fn
        img = Image.fromarray(new_arr)
        img.save(new_img_path)


if __name__ == "__main__":
    import sys

    for im in sys.argv[1:]:
        main(im)
