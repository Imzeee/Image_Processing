import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


def create_hisotgram(image_source_flat):
    histogram_array = np.zeros(256)
    for pixel in image_source_flat:
        histogram_array[pixel] += 1

    return histogram_array


def cumulative_sum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def calculate_cumulative_distribution_function(histogram):
    cdf = cumulative_sum(histogram)
    normalized_cdf = cdf / cdf[-1]
    return normalized_cdf


def calculate_lookup(source_cdf, reference_cdf):
    lookup_table = np.zeros(256)
    classes = 9
    n = 0

    for index in range(256):
        if source_cdf[index] >= reference_cdf[round(255 / classes * (n + 1))]:
            n += 1
            if n >= classes:
                n = classes - 1
        lookup_table[index] = int((255 * n) / (classes - 1))

    return lookup_table


def mono_histograms_matching(source_image, reference_histogram):
    source_histogram = create_hisotgram(source_image.flatten())
    reference_cdf = calculate_cumulative_distribution_function(reference_histogram)
    source_cdf = calculate_cumulative_distribution_function(source_histogram)

    lookup_table = calculate_lookup(source_cdf, reference_cdf)
    matched_image = []

    for y in source_image:
        for x in y:
            matched_image.append(lookup_table[x])

    matched_image_result = np.array(matched_image, dtype=np.uint8)

    return matched_image_result


def rgb_histograms_matching(source_image, reference_histogram):
    source_red = source_image[:, :, 0]
    source_green = source_image[:, :, 1]
    source_blue = source_image[:, :, 2]

    source_histogram_red = create_hisotgram(source_red.flatten())
    source_histogram_green = create_hisotgram(source_green.flatten())
    source_histogram_blue = create_hisotgram(source_blue.flatten())

    src_cdf_red = calculate_cumulative_distribution_function(source_histogram_red)
    src_cdf_green = calculate_cumulative_distribution_function(source_histogram_green)
    src_cdf_blue = calculate_cumulative_distribution_function(source_histogram_blue)
    ref_cdf = calculate_cumulative_distribution_function(reference_histogram)

    lut_red = calculate_lookup(src_cdf_red, ref_cdf)
    lut_green = calculate_lookup(src_cdf_green, ref_cdf)
    lut_blue = calculate_lookup(src_cdf_blue, ref_cdf)

    result = np.array([[[lut_red[r], lut_green[g], lut_blue[b]] for r, g, b in x] for x in source_image],
                      dtype=np.uint8)

    return result


def my_sort(part, sort_order):
    my_list = list(part.flatten())
    sorted_list = sorted(my_list)
    return sorted_list[sort_order - 1]


def ordfilt2(source_image, order, structuring_element):
    mask_x = structuring_element.shape[0]
    mask_y = structuring_element.shape[1]
    img_width = source_image.shape[0]
    img_height = source_image.shape[1]
    mask_x_half = math.floor(mask_x / 2)
    mask_y_half = math.floor(mask_y / 2)
    result_img = np.zeros_like(source_image)

    for i in range(math.ceil(mask_x / 2), img_width - math.floor(mask_x / 2)):
        for j in range(math.ceil(mask_y / 2), img_height - mask_y_half):
            part = source_image[i - mask_x_half:i + mask_x_half + 1, j - mask_y_half: j + mask_y_half + 1]
            index = (structuring_element == 1)
            result_img[i, j] = my_sort(part[index], order)

    return result_img


def disk_shape(radius):
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    mask_int = mask.astype(int)
    return mask_int


def imclose(source_image, radius):
    structuring_element = disk_shape(radius)

    padd = structuring_element.shape[0]
    padding = padd - 2
    image_padded = np.pad(source_image, pad_width=padding)

    result_image = erode(dilate(image_padded, structuring_element), structuring_element)

    return result_image[padding + 2:-padding - 2, padding + 2:-padding - 2]


def erode(source_image, structuring_element):
    mask_x = structuring_element.shape[0]
    mask_y = structuring_element.shape[1]
    img_width = source_image.shape[0]
    img_height = source_image.shape[1]
    mask_x_half = math.floor(mask_x / 2)
    mask_y_half = math.floor(mask_y / 2)
    result_image = np.zeros_like(source_image)

    for i in range(math.ceil(mask_x / 2), img_width - math.floor(mask_x / 2)):
        for j in range(math.ceil(mask_y / 2), img_height - mask_y_half):
            part = source_image[i - mask_x_half:i + mask_x_half + 1, j - mask_y_half: j + mask_y_half + 1]
            index = (structuring_element == 1)
            result_image[i, j] = min(part[index])

    return result_image


def dilate(source_image, structuring_element):
    mask_x = structuring_element.shape[0]
    mask_y = structuring_element.shape[1]
    img_width = source_image.shape[0]
    img_height = source_image.shape[1]
    mask_x_half = math.floor(mask_x / 2)
    mask_y_half = math.floor(mask_y / 2)
    result_image = np.zeros_like(source_image)

    for i in range(math.ceil(mask_x / 2), img_width - math.floor(mask_x / 2)):
        for j in range(math.ceil(mask_y / 2), img_height - mask_y_half):
            part = source_image[i - mask_x_half:i + mask_x_half + 1, j - mask_y_half: j + mask_y_half + 1]
            index = (structuring_element == 1)
            result_image[i, j] = max(part[index])

    return result_image


def convex_hull(source_image):
    comparison_image = np.zeros_like(source_image)
    output_image = source_image

    degree_0_mask = [[1, 1, 0], [1, -1, 0], [1, 0, -1]]
    degree_45_mask = [[1, 1, 1], [1, -1, 0], [0, -1, 0]]
    structuring_element_0_degree = np.array(degree_0_mask, dtype=np.int32)
    structuring_element_45_degree = np.array(degree_45_mask, dtype=np.int32)

    while not np.array_equal(output_image, comparison_image):
        comparison_image = output_image
        for x in range(4):
            output_image = output_image | hit_miss(output_image, structuring_element_0_degree)
            output_image = output_image | hit_miss(output_image, structuring_element_45_degree)
            structuring_element_0_degree = np.rot90(structuring_element_0_degree)
            structuring_element_45_degree = np.rot90(structuring_element_45_degree)

    return output_image


def hit_miss(source_image, structuring_element):
    true_mask = structuring_element * (structuring_element == 1)
    false_mask = structuring_element * (structuring_element == -1) * -1
    res_image = erode(source_image, true_mask) & erode(~source_image, false_mask)

    return res_image


if __name__ == '__main__':
    print("1 - Aligning histogram to normal distribution with given standard deviation")
    print("2 - Ordfilt2 for given mask and order number")
    print("3 - Closing operation with disk-shape element for given radius")
    print("4 - Convex Hull")
    number = int(input("Choose number of the transformation: "))

    if number == 1:
        image_name = input("Type path to image: ")
        std = float(input("Specify standard deviation: "))
        image_type = input("Write type of an image [MONO/RGB]: ")

        if image_type == "MONO":

            img = Image.open(image_name)
            plot5 = plt.figure(5)
            plt.imshow(img, cmap='gray')
            plt.title("Obraz wejsciowy")
            plt.savefig('gauss_res1.png')

            normal_distribution = np.zeros(256)
            for i in range(256):
                normal_distribution[i] = math.exp(-(i / 255 - 0.5) ** 2 / (2 * std * std)) / 10

            image = np.asarray(img)
            flat = image.flatten()
            hist_wejsciowy = create_hisotgram(image)

            # DYSTRYBUANTY
            cs = calculate_cumulative_distribution_function(hist_wejsciowy)
            gauss_cs = calculate_cumulative_distribution_function(normal_distribution)
            plot2 = plt.figure(2)
            plt.plot(cs, label='wejsciowy')
            plt.plot(gauss_cs, label='gauss')

            plt.xlim([0, 255])
            plt.title("Dystrybuanta")
            plt.legend()
            plt.savefig('gauss_res2.png')

            # MATCH MONO
            matched = mono_histograms_matching(image, normal_distribution)
            flat_matched = matched.flatten()
            plot3 = plt.figure(3)
            plt.hist(flat_matched, bins=256)
            plt.title("Histogram obrazu wyjsciowego")
            img_new = np.reshape(flat_matched, image.shape)
            plt.savefig('gauss_res3.png')

            plot4 = plt.figure(4)
            plt.imshow(img_new, cmap='gray')
            plt.title("Obraz wyjsciowy")
            plt.savefig('gauss_res4.png')
            plt.show()
        else:

            img = Image.open(image_name)
            plot1 = plt.figure(5)
            plt.imshow(img)
            plt.title("Obraz wejsciowy")
            plt.savefig('gauss_rgb_res1.png')

            normal_distribution = np.zeros(256)
            for i in range(256):
                normal_distribution[i] = math.exp(-(i / 255 - 0.5) ** 2 / (2 * std * std)) / 10

            image = np.asarray(img)
            flat = image.flatten()
            hist_wejsciowy_r = create_hisotgram(image[:, :, 0])
            hist_wejsciowy_g = create_hisotgram(image[:, :, 1])
            hist_wejsciowy_b = create_hisotgram(image[:, :, 2])

            # DYSTRYBUANTY
            cs_r = calculate_cumulative_distribution_function(hist_wejsciowy_r)
            cs_g = calculate_cumulative_distribution_function(hist_wejsciowy_g)
            cs_b = calculate_cumulative_distribution_function(hist_wejsciowy_b)
            gauss_cs = calculate_cumulative_distribution_function(normal_distribution)

            plot2 = plt.figure(2)
            plt.plot(cs_r, label='wejsciowy_r', color="red")
            plt.plot(cs_g, label='wejsciowy_g', color="green")
            plt.plot(cs_b, label='wejsciowy_b', color="blue")
            plt.plot(gauss_cs, label='gauss', color="black")
            plt.legend()
            plt.xlim([0, 255])
            plt.title("Dystrybuanta")
            plt.savefig('gauss_rgb_res2.png')

            # MATCH RGB

            matched = rgb_histograms_matching(image, normal_distribution)
            plot6 = plt.figure(6)
            plt.hist(matched[:, :, 0].flatten(), bins=256, color="red")
            plt.title("Histogram obrazu wyjsciowego_RED")
            plt.savefig('gauss_rgb_res3.png')

            plot7 = plt.figure(7)
            plt.hist(matched[:, :, 1].flatten(), bins=256, color="green")
            plt.title("Histogram obrazu wyjsciowego_GREEN")
            plt.savefig('gauss_rgb_res4.png')

            plot8 = plt.figure(8)
            plt.hist(matched[:, :, 2].flatten(), bins=256, color="blue")
            plt.title("Histogram obrazu wyjsciowego_BLUE")
            plt.savefig('gauss_rgb_res5.png')

            plot4 = plt.figure(4)
            plt.imshow(matched)
            plt.title("Obraz wyjsciowy")
            plt.savefig('gauss_rgb_res6.png')
            plt.show()

    if number == 2:
        image_name = input("Type path to image: ")
        image_type = input("Write type of an image [MONO/RGB]: ")
        order = int(input("Specify order: "))
        mask_size = int(input("Specify size of the mask: "))
        mask = np.ones((mask_size, mask_size))

        if image_type == "MONO":
            img = Image.open(image_name)
            image = np.asarray(img)
            plot1 = plt.figure(1)
            res = ordfilt2(image, order, mask)
            plt.imshow(res, cmap='gray')
            plt.savefig('ordfilt2_res1.png')
            plot2 = plt.figure(2)
            plt.imshow(img, cmap='gray')
            plt.savefig('ordfilt2_res2.png')
            plt.show()
        else:
            img = Image.open(image_name)
            image = np.asarray(img)
            plot1 = plt.figure(1)
            result_image = np.zeros_like(image)
            result_image[:, :, 0] = ordfilt2(image[:, :, 0], order, mask)
            result_image[:, :, 1] = ordfilt2(image[:, :, 1], order, mask)
            result_image[:, :, 2] = ordfilt2(image[:, :, 2], order, mask)
            plt.imshow(result_image)
            plt.savefig('ordfilt2_rgb_res1.png')
            plot2 = plt.figure(2)
            plt.imshow(img)
            plt.savefig('ordfilt2_rgb_res2.png')
            plt.show()

    if number == 3:
        image_name = input("Type path to image: ")
        radius = int(input("Specify radius: "))
        image_type = input("Write type of an image [MONO/LOGICAL]: ")

        if image_type == "MONO":
            img = Image.open(image_name)
            image = np.asarray(img)
            plot2 = plt.figure(2)
            plt.imshow(img, cmap='gray')
            plt.savefig('closing_mono_res1.png')

            plot1 = plt.figure(1)
            res = imclose(image, radius)
            plt.imshow(res, cmap='gray')
            plt.savefig('closing_mono_res2.png')
            plt.show()
        else:
            img = Image.open(image_name).convert('L')
            image = np.asarray(img)
            plot2 = plt.figure(2)
            plt.imshow(img, cmap='gray')
            plt.savefig('closing_logical_res1.png')

            plot1 = plt.figure(1)
            res = imclose(image, radius)
            plt.imshow(res, cmap='gray')
            plt.savefig('closing_logical_res2.png')
            plt.show()

    if number == 4:
        image_name = input("Type path to image: ")
        img = Image.open(image_name).convert('L')

        image = np.asarray(img, dtype=int)

        res = convex_hull(image)
        plot1 = plt.figure(1)
        plt.imshow(res, cmap='gray')
        plt.savefig('convex_hull_res1.png')

        plot2 = plt.figure(2)
        plt.imshow(img, cmap='gray')
        plt.savefig('convex_hull_res2.png')

        plt.show()
