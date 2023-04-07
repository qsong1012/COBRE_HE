import openslide
import numpy as np


class WSI:
    """
    using openslide
    """

    def __init__(self, svs_path):
        self.svs_path = svs_path
        self.slide = openslide.OpenSlide(svs_path)
        self.mag_ori = int(
            float(self.slide.properties.get('aperio.AppMag', 40)))

    def get_region(self, x, y, size, mag, mag_mask):
        dsf = self.mag_ori / mag
        level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][level])
        dsf = mag_new / mag
        dsf_mask = self.mag_ori / mag_mask
        img = self.slide.read_region((int(x * dsf_mask), int(y * dsf_mask)),
                                     level, (int(size * dsf), int(size * dsf)))
        return np.array(img.convert('RGB').resize((size, size)))

    def downsample(self, mag):
        dsf = self.mag_ori / mag
        level = self.get_best_level_for_downsample(dsf)
        mag_new = self.mag_ori / (
            [int(x) for x in self.slide.level_downsamples][level])
        dsf_new = self.mag_ori / mag_new
        img = self.slide.read_region(
            (0, 0), level,
            tuple(int(x / dsf_new) for x in self.slide.dimensions))
        sizes = tuple(int(x // dsf) for x in self.slide.dimensions)
        return np.array(img.convert('RGB').resize(sizes))

    def get_best_level_for_downsample(self, factor):
        levels = [int(x) for x in self.slide.level_downsamples]

        for i, level in enumerate(levels):
            if factor == level:
                return i
            elif factor > level:
                continue
            elif factor < level:
                return max(i - 1, 0)

        return len(levels) - 1