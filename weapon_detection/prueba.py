import detecting_images


if __name__ == "__main__":
    detector = detecting_images.WeaponDetector()
    # Example usage:
    detector.detect_objects_in_photo("./dataset/input/armas (98).jpg")
