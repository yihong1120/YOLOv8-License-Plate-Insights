import glob
import os
import imageio.v2 as imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa

class DataAugmentation:
    def __init__(self, train_path, num_augmentations=1):
        self.train_path = train_path
        self.num_augmentations = num_augmentations
        self.seq = iaa.Sequential([
            iaa.Flipud(0.5),  # 垂直翻转图像和标签
            iaa.Fliplr(0.5),  # 水平翻转图像和标签
            iaa.Affine(rotate=(-25, 25)),  # 旋转图像和标签
            iaa.Multiply((0.8, 1.2)),  # 改变亮度，不改变颜色
            iaa.LinearContrast((0.75, 1.5)),  # 改变对比度
            iaa.AddToHueAndSaturation((-20, 20)),  # 改变色调和饱和度
            iaa.GaussianBlur(sigma=(0, 0.5)),  # 应用高斯模糊
            iaa.Grayscale(alpha=(0.0, 1.0)),  # 部分转换为灰度图像
            iaa.Resize((0.5, 1.5)),
            iaa.Crop(px=(0, 16)),
            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                            children=iaa.WithChannels(0, iaa.Add((10, 50)))),
            iaa.SaltAndPepper(0.05),
            iaa.ElasticTransformation(alpha=50, sigma=5),
            iaa.Superpixels(p_replace=0.5, n_segments=64),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.PiecewiseAffine(scale=(0.01, 0.05))
        ], random_order=True)

    def augment_data(self):
        image_paths = glob.glob(os.path.join(self.train_path, 'images', '*.png'))

        for image_path in image_paths:
            image = imageio.imread(image_path)
            label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
            image_shape = image.shape
            bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

            for i in range(self.num_augmentations):
                if image.shape[2] == 4:
                    image = image[:, :, :3]

                image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)

                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                aug_image_filename = f"{base_filename}_aug_{i}.png"
                aug_label_filename = f"{base_filename}_aug_{i}.txt"
                
                image_aug_path = os.path.join(self.train_path, 'images', aug_image_filename)
                label_aug_path = os.path.join(self.train_path, 'labels', aug_label_filename)

                imageio.imwrite(image_aug_path, image_aug)
                self.write_label_file(bbs_aug.remove_out_of_image().clip_out_of_image(), label_aug_path, image_shape[1], image_shape[0])

    @staticmethod
    def read_label_file(label_path, image_shape):
        bounding_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * image_shape[1]
                    y1 = (y_center - height / 2) * image_shape[0]
                    x2 = (x_center + width / 2) * image_shape[1]
                    y2 = (y_center + height / 2) * image_shape[0]
                    bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
        return bounding_boxes

    @staticmethod
    def write_label_file(bounding_boxes, label_path, image_width, image_height):
        with open(label_path, 'w') as f:
            for bb in bounding_boxes:
                x_center = ((bb.x1 + bb.x2) / 2) / image_width
                y_center = ((bb.y1 + bb.y2) / 2) / image_height
                width = (bb.x2 - bb.x1) / image_width
                height = (bb.y2 - bb.y1) / image_height
                class_index = bb.label
                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    train_path = 'dataset/train'
    num_augmentations = 2
    augmenter = DataAugmentation(train_path, num_augmentations)
    augmenter.augment_data()