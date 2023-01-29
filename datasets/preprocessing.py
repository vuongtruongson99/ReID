import random
import math

class RandomErasing(object):
    """ Chọn ngẫu nhiên 1 vùng hình chữ nhật trong 1 ảnh và xóa các pixels trong nó

    Args:
        probability: Xác suất để phép xóa ngẫu nhiên được thực hiện
        sl: Tỷ lệ tối thiểu của khu vực bị xóa so với input img
        sh: Tỷ lệ tối đa của khu vực bị xóa so với input img
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img