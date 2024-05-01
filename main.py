import argparse
import math
from PIL import Image
import numpy as np
from numba import cuda



# Kernel to convert an RGB image to gray scale
@cuda.jit
def kernel2D_writeBWArray(d_src, output_array):
    x, y = cuda.grid(2)
    if x < d_src.shape[0] and y < d_src.shape[1]:  # check that threads don't refer to invalid location
        r, g, b = d_src[x, y]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # computation of the level of gray
        output_array[x, y] = (gray, gray, gray)


# Kernel to apply gaussian filter on the input_image, as effect the image will be blurred
@cuda.jit
def gaussian_filter_kernel2D(input_image, output_array, kernel):
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:  # check that threads don't refer to invalid location
        radius = kernel.shape[0] // 2
        pixel_value = 0.0
        for i in range(-radius, radius + 1):  # slide through the gaussian mask
            for j in range(-radius, radius + 1):
                pixel_value += input_image[x - i, y - j][2] * kernel[
                    i + radius, j + radius]  # computation of new pixel value
        output_array[x, y] = pixel_value


# Kernel that apply the horizontal and vertical sobel filter to the input_image
@cuda.jit
def sobel_filter_kernel2D(input_image, output_mag):
    x, y = cuda.grid(2)

    if x < input_image.shape[0] and y < input_image.shape[1]: # check that threads don't refer to invalid location
        if 0 < x < input_image.shape[0] - 1 and 0 < y < input_image.shape[1] - 1:
            #computaion of horizontal enhancement
            gx = input_image[x + 1, y - 1][2] + 2 * input_image[x + 1, y][2] + input_image[x + 1, y + 1][2] - \
                 input_image[x - 1, y - 1][2] - 2 * input_image[x - 1, y][2] - input_image[x - 1, y + 1][2]
            #computaion of vertical enhancement
            gy = input_image[x - 1, y + 1][2] + 2 * input_image[x, y + 1][2] + input_image[x + 1, y + 1][2] - \
                 input_image[x - 1, y - 1][2] - 2 * input_image[x, y - 1][2] - input_image[x + 1, y - 1][2]

            magnitude = min(int(math.ceil((gx ** 2 + gy ** 2) ** 0.5)), 175) #clamping to 175 the magnitude value
            output_mag[x, y] = magnitude




# Kernel to classify the pixel as strong (sure in edge), weak (maybe), non relevant
@cuda.jit
def threshold_kernel2D(img, output_array):
    x, y = cuda.grid(2)
    highThreshold = 102
    lowThreshold = 51
    weak = 127
    strong = 255
    if x < img.shape[0] and y < img.shape[1]:  # check that threads don't refer to invalid location
        # classification based on the high and low threshold
        if img[x, y][2] >= highThreshold:
            output_array[x, y] = strong
        elif img[x, y][2] >= lowThreshold:
            output_array[x, y] = weak
        else:
            output_array[x, y] = 0


# Kernel that transform weak point in strong if in their neighbourhood there is at leat one strong point
@cuda.jit
def hysteresis_kernel2D(img, output_array):
    weak = 127
    strong = 255
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:  # check that threads don't refer to invalid location
        if img[x, y][2] == weak:  # identify of weak point
            if ((img[x + 1, y - 1][2] == strong) or (img[x + 1, y][2] == strong) or (
                    img[x + 1, y + 1][2] == strong)  # check of the neighbourhood
                    or (img[x, y - 1][2] == strong) or (img[x, y + 1][2] == strong)
                    or (img[x - 1, y - 1][2] == strong) or (img[x - 1, y][2] == strong) or (
                            img[x - 1, y + 1][2] == strong)):
                output_array[x, y] = strong  # strong pixel found, the starting pixel from weak become strong
            else:
                output_array[x, y] = 0
        else:
            output_array[x, y] = img[x, y][2]


# function that instance in correct way the number of blocks per grid base on the number of threads per block
def create_blocks(input_array, tpb):
    threads_per_block = (tpb, tpb)
    num_blocks_x = (input_array.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    num_blocks_y = (input_array.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    block_dim_x = min(input_array.shape[0], num_blocks_x)
    block_dim_y = min(input_array.shape[1], num_blocks_y)
    blocks_per_grid = (block_dim_x, block_dim_y)
    return blocks_per_grid


def apply_bw(input_image, tpb):
    input_image = np.array(input_image)
    d_In = cuda.to_device(input_image)
    d_Out = cuda.device_array_like(d_In)
    threads_per_block = (tpb, tpb)
    blocks_per_grid = create_blocks(input_image, tpb)
    kernel2D_writeBWArray[blocks_per_grid, threads_per_block](d_In, d_Out)
    out = np.array(input_image)
    d_Out.copy_to_host(out)
    return out


def create_blurred_image(input_image, tpb):
    input_image = np.array(input_image)
    d_In = cuda.to_device(input_image)
    d_Out = cuda.device_array_like(d_In)
    kernel: np.ndarray = np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]], dtype=np.float32) / 256.0

    d_K = cuda.to_device(kernel)
    threads_per_block = (tpb, tpb)
    blocks_per_grid = create_blocks(input_image, tpb)
    gaussian_filter_kernel2D[blocks_per_grid, threads_per_block](d_In, d_Out, d_K)
    out = np.array(input_image)
    d_Out.copy_to_host(out)
    return out


def apply_sobel(input_image, tpb):
    input_image = np.array(input_image)
    d_In = cuda.to_device(input_image)
    d_Mag = cuda.device_array_like(d_In)
    threads_per_block = (tpb, tpb)
    blocks_per_grid = create_blocks(input_image, tpb)
    sobel_filter_kernel2D[blocks_per_grid, threads_per_block](d_In, d_Mag)

    mag = np.array(input_image)
    d_Mag.copy_to_host(mag)
    return mag


def apply_threshold_kernel(img, tpb):
    input_img = np.array(img)
    d_In = cuda.to_device(img)
    d_Out = cuda.device_array_like(d_In)
    threads_per_block = (tpb, tpb)
    blocks_per_grid = create_blocks(input_img, tpb)
    threshold_kernel2D[blocks_per_grid, threads_per_block](d_In, d_Out)
    out = np.array(img)
    d_Out.copy_to_host(out)
    return out


def apply_hysteresis(img, tpb):
    d_In = cuda.to_device(img)
    d_Out = cuda.device_array_like(d_In)
    threads_per_block = (tpb, tpb)
    blocks_per_grid = create_blocks(img, tpb)
    hysteresis_kernel2D[blocks_per_grid, threads_per_block](d_In, d_Out)
    out = np.array(img)
    d_Out.copy_to_host(out)
    return out


# function that will compute all the steps to return the final image
def do_everything(input_img, output_img, thread_per_block):
    src = np.array(input_img)

    gray_pil = apply_bw(src, thread_per_block)
    gray_pil_img = Image.fromarray(gray_pil)
    #gray_pil_img.save("bw.jpg")

    blurred_image = create_blurred_image(src, thread_per_block)
    blurred_img = Image.fromarray(blurred_image)
    #blurred_img.save("blurred.jpg")

    magnitude = apply_sobel(blurred_image, thread_per_block)
    sobel_img = Image.fromarray(magnitude)
    #sobel_img.save("sobel.jpg")

    thresholded = apply_threshold_kernel(magnitude, thread_per_block)
    thresholded_img = Image.fromarray(thresholded)
    #thresholded_img.save("thresholded.jpg")

    hysteresised = apply_hysteresis(thresholded, thread_per_block)
    hysteresised_img = Image.fromarray(hysteresised)
    hysteresised_img.save(output_img)


def main():
    out_name = "output.jpg"
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb', type=int, default=1)
    parser.add_argument('--bw', action='store_true')
    parser.add_argument('--gauss', action='store_true')
    parser.add_argument('--sobel', action='store_true')
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument('inputImage', type=str)
    parser.add_argument('outputImage', type=str)
    args = parser.parse_args()

    img = Image.open(args.inputImage)
    out_name = args.outputImage
    thread_per_block = 16
    if args.tb:
        if args.tb and 1 <= args.tb <= 32:
            thread_per_block = args.tb
        else:
            print(
                "The number of threads per block must be between 1 and 32, inclusive. The default value (16) will be used.")
            thread_per_block = 16
    if args.bw:
        output = apply_bw(img, thread_per_block)
        out_img = Image.fromarray(output)
        out_img.save(out_name)
    if args.gauss:
        output = apply_bw(img, thread_per_block)
        out_img = Image.fromarray(output)

        output = create_blurred_image(out_img, thread_per_block)
        out_img = Image.fromarray(output)
        out_img.save(out_name)

    if args.sobel:
        output = apply_bw(img, thread_per_block)
        out_img = Image.fromarray(output)

        output = create_blurred_image(out_img, thread_per_block)
        out_img = Image.fromarray(output)

        output_mag = apply_sobel(out_img, thread_per_block)
        out_img = Image.fromarray(output_mag)
        out_img.save(out_name)

    if args.threshold:
        src = np.array(img)

        gray_pil = apply_bw(src, thread_per_block)
        gray_pil_img = Image.fromarray(gray_pil)

        blurred_image = create_blurred_image(src, thread_per_block)
        blurred_img = Image.fromarray(blurred_image)

        magnitude = apply_sobel(blurred_image, thread_per_block)
        sobel_img = Image.fromarray(magnitude)

        thresholded = apply_threshold_kernel(magnitude, thread_per_block)
        thresholded_img = Image.fromarray(thresholded)
        thresholded_img.save(out_name)

    if not (args.bw or args.gauss or args.sobel or args.threshold):
        do_everything(img, args.outputImage, thread_per_block)


if __name__ == '__main__':
    main()