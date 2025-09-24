import glob
import os

def mktxt() -> None:
    train_image_paths = sorted(glob.glob(os.path.join("**", "train", "*.jpg")))
    val_image_paths = sorted(glob.glob(os.path.join("**", "valid", "*.jpg")))
    test_image_paths = sorted(glob.glob(os.path.join("**", "test", "*.jpg")))

    for subset, image_paths in [["train", train_image_paths], ["valid", valid_image_paths], ["test", test_image_paths]]:
            with open(f"{subset}.txt", "w") as txt_file:
                for image_path in image_paths:
                    txt_file.write(f"{os.path.abspath(image_path)}\n")

if __name__ == "__main__":
    mktxt()
