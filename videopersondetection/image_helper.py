from PIL import Image

PADDING = 5


class ImageHelper:
    def create_from_frame(self, frame):
        return Image.fromarray(frame).convert('RGB')

    def crop(self, image, bounding_box):
        return image.crop((bounding_box.xmin - PADDING, bounding_box.ymin - PADDING, bounding_box.xmax + PADDING, bounding_box.ymax + PADDING))
