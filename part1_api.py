from numpy import float32

try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def get_kernel_circle(kernel_size):
    """Get the desired size, open the relevant image and create a kernel from it"""
    circle_img = Image.open(f'{kernel_size}Kernel.png')
    kernel_circle = np.asarray(circle_img)[:, :, 0]  # take one layer
    kernel_circle = kernel_circle.astype(float32)
    kernel_circle -= 100  # The black dots will be negative and the white would remain positive.
    sum_circle = np.sum(kernel_circle)
    # Calculate the size and normal the kernel to sum of one.
    area = circle_img.width * circle_img.height
    kernel_circle -= (sum_circle / area)
    max_kernel_circle = np.max(kernel_circle)
    kernel_circle /= max_kernel_circle
    return kernel_circle


# def filter_white_dots(c_image):
#     white_dots_filter = np.array([True for _ in range(c_image.size) for _ in range(c_image[0].size)])
#     for i in range(c_image.size):
#         for j in range(c_image[0].size):
#             if all([c_image[i, j, 0] > 250, c_image[i, j, 1] > 250, c_image[i, j, 2] > 250]):
#                 white_dots_filter[i, j] = False
#     return white_dots_filter


def get_dots(image, threshold):
    """Get a mat of dots and threshold and return the filtered dots."""
    image_converted = image.astype(np.float32)[10:-5, 10:-10]
    filtered = maximum_filter(image_converted, 30)
    lights_indices = np.argwhere((threshold < image_converted) & (filtered == image_converted))
    return lights_indices + 10


def find_tfl_by_layer(c_image, layer, t_small, t_big):
    """Get an image, layer and the threshold and return the attention dots. """
    c_image_bw = Image.fromarray(c_image)
    c_image_bw_array = np.asarray(c_image_bw)[:, :, layer]
    c_image_bw_array = c_image_bw_array.astype(float32)

    # find_small_tfl
    kernel_circle = get_kernel_circle("small")
    res = sg.convolve(c_image_bw_array, kernel_circle, mode='same', method='auto')
    threshold = np.max(res[10:-5, 10:-10]) - t_small  # Filter the frame dots.
    dots_small = get_dots(res, threshold)

    # find_big_tfl
    kernel_circle = get_kernel_circle("big")
    res = sg.convolve(c_image_bw_array, kernel_circle, mode='same', method='auto')
    threshold = np.max(res[10:-5, 10:-10]) - t_big  # Filter the frame dots.
    dots_big = get_dots(res, threshold)

    dots = np.vstack([dots_small, dots_big])
    return dots


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    dots_red_layer = find_tfl_by_layer(c_image, 0, 1000, 5000)
    dots_green_layer = find_tfl_by_layer(c_image, 1, 1000, 5000)
    # Filter the attention dots to Green and Red.
    filtered_red_dots, filtered_green_dots = np.array([[0, 0]]), np.array([[0, 0]])
    for dot in dots_red_layer:
        dot_color = c_image[dot[0], dot[1]]
        if dot_color[0] >= dot_color[1]:
            filtered_red_dots = np.append(filtered_red_dots, [[dot[0], dot[1]]], axis=0)
    for dot in dots_green_layer:
        dot_color = c_image[dot[0], dot[1]]
        if dot_color[1] > dot_color[0]:
            filtered_green_dots = np.append(filtered_green_dots, [[dot[0], dot[1]]], axis=0)
    return filtered_red_dots[1:, 1], filtered_red_dots[1:, 0], filtered_green_dots[1:, 1], filtered_green_dots[1:, 0]


def show_image_and_gt(image, objs, fig_num=None):
    """Call to the finders functions and show the image with Green and Red dots on that. """
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'try'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
