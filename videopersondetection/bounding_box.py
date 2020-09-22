import collections


class BoundingBox(collections.namedtuple('BoundingBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    __slots__ = ()

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        xcenter = self.xmin + int(self.width / 2.0)
        ycenter = self.ymin + int(self.height / 2.0)
        return (xcenter, ycenter)

    def scale(self, sx, sy):
        return BoundingBox(
            xmin=sx * self.xmin,
            ymin=sy * self.ymin,
            xmax=sx * self.xmax,
            ymax=sy * self.ymax)

    def map(self, f):
        return BoundingBox(
            xmin=f(self.xmin),
            ymin=f(self.ymin),
            xmax=f(self.xmax),
            ymax=f(self.ymax))
