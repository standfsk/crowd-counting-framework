import os
from pathlib import Path

def mktxt() -> None:
    input_path = "."
    train_image_paths = sorted(Path(input_path).glob("*/train/*.jpg"))
    valid_image_paths = sorted(Path(input_path).glob("*/valid/*.jpg"))
    test_image_paths = sorted(Path(input_path).glob("*/test/*.jpg"))

    for subset, image_paths in [["train", train_image_paths], ["valid", valid_image_paths], ["test", test_image_paths]]:
            with open(f"{subset}.txt", "w") as txt_file:
                for image_path in image_paths:
                    txt_file.write(f"{os.path.abspath(image_path)}\n")

if __name__ == "__main__":
    mktxt()
