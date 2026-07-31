from config import TEST_IMAGES_DIR
from data_loader import preprocess_image


def main():

    image_path = TEST_IMAGES_DIR / "39.jpg"

    image = preprocess_image(image_path)

    print("=" * 50)
    print("Image preprocessing successful")
    print("=" * 50)

    print(f"Shape : {image.shape}")
    print(f"Type  : {image.dtype}")
    print(f"Min   : {image.min():.3f}")
    print(f"Max   : {image.max():.3f}")


if __name__ == "__main__":
    main()