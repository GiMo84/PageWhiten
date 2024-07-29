import os
import cv2
import numpy as np
import click
import click_logging
import logging
import inspect
from functools import wraps

# Initialize logging
logger = logging.getLogger(__name__)
click_logging.basic_config(logger)


def pass_var_names_and_values(func):
    """
    Decorator function that passes variable names and values to the decorated function.

    Examples:
        >>> @pass_var_names_and_values
        ... def my_function(arg1, arg2, **kwargs):
        ...     print(arg1, arg2)
        ...     print(kwargs[var_names_dict])
        ...
        >>> x = 1
        >>> y = 2
        >>> my_function(x, y)
        1 2
        {'arg1': 'x', 'arg2': 'y'}

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the previous frame in the call stack, which is the caller of this function
        caller_frame = inspect.currentframe().f_back
        # Get the function's argument names
        func_args = inspect.getfullargspec(func).args
        
        # Map the arguments to their names
        arg_dict = {name: value for name, value in zip(func_args, args)}
        
        # Update with kwargs (keyword arguments)
        arg_dict.update(kwargs)
        
        # Get the local variables from the caller's frame
        caller_locals = caller_frame.f_locals
        
        # Create a dictionary with variable names as keys and (name, value) as values
        var_name_value_dict = {}
        for name, value in arg_dict.items():
            for var_name, var_value in caller_locals.items():
                if var_value is value:
                    var_name_value_dict[name] = (var_name, value)
                    break
            else:
                var_name_value_dict[name] = (None, value)
        
        return func(**{k: v for k, (_, v) in var_name_value_dict.items()}, var_names_dict = {k: n for k, (n, _) in var_name_value_dict.items()})
    
    return wrapper

@pass_var_names_and_values
def save_image_step(output_path_noext, step, output_ext, image, **kwargs):
    """
    Save an image to a file with a specific name and extension. Takes the step name from the name of the image variable.

    Args:
        output_path_noext (str): The path to the output file without the file extension.
        step (int): The step number used in the file name.
        output_ext (str): The file extension to be used.
        image (numpy.ndarray): The image to be saved.

    Examples:
        >>> save_image("output", 0, ".jpg", image_foo)
        (saves output_00_foo.jpg)

    """
    step_name = kwargs['var_names_dict']['image']
    cv2.imwrite(f"{output_path_noext}_{step:02d}_{step_name}{output_ext}", image)

def mask_from_color(image, blur_kernel_size=15, threshold_block_size=15, threshold_C=-2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_mask = cv2.adaptiveThreshold(
        cv2.GaussianBlur(hsv[:, :, 1], (blur_kernel_size, blur_kernel_size), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        threshold_block_size,
        threshold_C
    )
    color_mask = cv2.bitwise_not(color_mask)
    return color_mask

def filter_lighter_color(image1, image2):
    img_out = np.maximum(image1, image2)
    return img_out

def filter_subtract(image1, image2):
    img_out = -(image1.astype(np.int16) - image2.astype(np.int16)).astype(np.int16)
    img_out[img_out < 0] = 0
    return img_out.astype(np.int8)

def combine_with_mask(image1, image2, mask):
    combined = cv2.bitwise_and(image1, image1, mask=mask) + cv2.bitwise_and(image2, image2, mask=cv2.bitwise_not(mask))
    return combined

@click.command()
@click.argument("input", type=click.Path(exists=True, file_okay=True), nargs=-1, required=True)
@click.option("--output", "-o", "output_path", default='.', type=click.Path(writable=True), help="Output file or path.")
@click.option("--gamma-tint-mask", "-gt", default=1.0, type=click.FLOAT, help="Gamma correction for the tint mask.")
@click.option("--gamma-sep", "-gs", default=1.0, type=click.FLOAT, help="Gamma correction for the foreground/background separation mask (not compatible with --onestep).")
@click.option('--denoise-tint-mask', nargs=4, default=(10, 10, 7, 21), type=click.Tuple([click.INT, click.INT, click.INT, click.INT]), help='Denoising parameters for the tint mask.')
@click.option('--denoise-bg', nargs=4, default=(-1, -1, -1, -1), type=click.Tuple([click.INT, click.INT, click.INT, click.INT]), help='Denoising parameters for the background. 0 disables denoising, -1 uses tint-mask denoising parameters.')
@click.option('--denoise-fg', nargs=4, default=(0, 0, 0, 0), type=click.Tuple([click.INT, click.INT, click.INT, click.INT]), help='Denoising parameters for the foreground. 0 disables denoising, -1 uses tint-mask denoising parameters.')
@click.option('--inpaint-bg', is_flag=True, help='Inpaint the background.')
@click.option('--kernel-size', default=5, type=click.INT, help='Kernel size for morphological operations.')
@click.option('--erode-iterations', default=1, type=click.INT, help='Number of iterations for the mask erosion operation.')
@click.option('--dilate-erode-iterations', default=100, type=click.INT, help='Number of iterations for the blurred background mask dilation and erosion operation.')
@click.option('--save-intermediate', '-i', is_flag=True, help='Flag to save intermediate images.')
@click.option('--separate', '-s', is_flag=True, help='Flag to save separately foreground and background images.')
@click.option('--onestep', is_flag=True, help='Flag to use one-step process (forces --inpaint-bg).')
@click_logging.simple_verbosity_option(logger)
def main(input, output_path, gamma_tint_mask, gamma_sep, denoise_tint_mask, denoise_bg, denoise_fg, inpaint_bg, kernel_size, erode_iterations, dilate_erode_iterations, save_intermediate, separate, onestep):
    """
    Removes the background tint and enhance the foreground of images.
    """

    if len(input) == 1:
        if os.path.isdir(output_path):
            fname = os.path.split(output_path)[1]
            output_file = os.path.join(output_path, os.path.splitext(fname)[0] + '_out' + os.path.splitext(fname)[1])
        else:
            output_file = output_path
    else:
        if not(os.path.isdir(output_path)):
            logger.error("Output must be a path if multiple input files are given.")
            raise NotADirectoryError
    for input_file in input:
        if len(input) > 1:
            fname = os.path.split(input_file)[1]
            output_file = os.path.join(output_path, os.path.splitext(fname)[0] + '_out' + os.path.splitext(fname)[1])
        click.echo(f"Processing {input_file} -> {output_file}")
        process_image(input_file, output_file, gamma_tint_mask, gamma_sep, denoise_tint_mask, denoise_bg, denoise_fg, inpaint_bg, kernel_size, erode_iterations, dilate_erode_iterations, save_intermediate, separate, onestep)

def process_image(image_path, output_path, gamma_tint_mask, gamma_sep, denoise_tint_mask, denoise_bg, denoise_fg, inpaint_bg, kernel_size, erode_iterations, dilate_erode_iterations, save_intermediate, separate, onestep):
    """
    Process an image by performing a series of operations to remove the background tint and enhance the foreground.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to save the processed image.
        gamma-tint-mask (float): Gamma correction for the tint mask.
        gamma-sep (float): Gamma correction for the foreground/background separation mask (not compatible with --onestep).
        denoise-tint-mask (tuple): Denoising parameters for the tint mask.
        denoise-bg (tuple): Denoising parameters for the background. 0 disables denoising, -1 uses tint-mask denoising parameters.
        denoise-fg (tuple): Denoising parameters for the foreground. 0 disables denoising, -1 uses tint-mask denoising parameters.
        inpaint-bg (bool): Inpaint the background.
        kernel-size (int): Kernel size for morphological operations.
        erode-iterations (int): Number of iterations for the mask erosion operation.
        dilate-erode-iterations (int): Number of iterations for the blurred background mask dilation and erosion operation.
        save-intermediate (bool): Flag to save intermediate images.
        separate (bool): Flag to save separately foreground and background images.
        onestep (bool): Flag to use one-step process (forces --inpaint-bg).
    """
    output_path_noext = os.path.splitext(output_path)[0]
    output_ext = os.path.splitext(output_path)[1]

    #logger.debug(f"Output path (no extension): {output_path_noext}")
    #logger.debug(f"Output extension: {output_ext}")

    denoise_h, denoise_h_for_color, denoise_template_window_size, denoise_search_window_size = denoise_tint_mask
    denoise_bg_h, denoise_bg_h_for_color, denoise_bg_template_window_size, denoise_bg_search_window_size = denoise_bg
    denoise_fg_h, denoise_fg_h_for_color, denoise_fg_template_window_size, denoise_fg_search_window_size = denoise_fg

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    step = 0

    logger.info("Reading the image")
    original_image = cv2.imread(image_path)
    
    step += 1
    if denoise_h != 0 or denoise_h_for_color != 0:
        logger.info("Denoising the image")
        denoised_image = cv2.fastNlMeansDenoisingColored(original_image, None, denoise_h, denoise_h_for_color, denoise_template_window_size, denoise_search_window_size)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, denoised_image)
    else:
        denoised_image = original_image

    logger.info("Converting image to grayscale")
    step += 1
    grayscale_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    if gamma_tint_mask != 1.0:
        grayscale_image_gamma_corr = (cv2.pow(grayscale_image.astype(np.double) / 255.0, gamma_tint_mask) * 255.0).astype(np.uint8)
    else:
        grayscale_image_gamma_corr = grayscale_image
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, grayscale_image_gamma_corr)

    logger.info("Applying Otsu's threshold")
    step += 1
    _, otsu_mask = cv2.threshold(grayscale_image_gamma_corr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, otsu_mask)

    logger.info("Enhancing the mask using color info")
    step += 1
    color_mask = mask_from_color(denoised_image)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, color_mask)

    logger.info("Combining masks")
    step += 1
    combined_mask = cv2.bitwise_and(otsu_mask, color_mask)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, combined_mask)

    logger.info("Eroding the mask")
    step += 1
    eroded_mask = cv2.erode(combined_mask, kernel, erode_iterations)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, eroded_mask)

    logger.info("Inpainting the background")
    step += 1
    inpainted_background = cv2.inpaint(denoised_image, cv2.bitwise_not(eroded_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, inpainted_background)

    logger.info("Blurring the background")
    step += 1
    blurred_background = cv2.dilate(cv2.blur(cv2.erode(cv2.medianBlur(cv2.add(inpainted_background, cv2.cvtColor(cv2.bitwise_not(combined_mask), cv2.COLOR_GRAY2BGR)), 255), kernel, iterations=dilate_erode_iterations), (255, 255)), kernel, iterations=dilate_erode_iterations)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, blurred_background)

    logger.info("Calculating background tint")
    step += 1
    # Get lighter color between inpainted and blurred
    tint_background = filter_lighter_color(blurred_background, inpainted_background)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, tint_background)
    
    if not onestep:
        step += 100
        if denoise_bg_h >= 0 or denoise_bg_h_for_color >= 0 or denoise_fg_h >= 0 or denoise_fg_h_for_color >= 0:
            logger.info("Removing the tint from the original image")
            image_no_tint = (np.int32(255) - filter_subtract(original_image, tint_background)).astype(np.uint8)
            if save_intermediate:
                save_image_step(output_path_noext, step, output_ext, image_no_tint)

        logger.info("Removing the tint from the denoised image")
        step += 1
        denoised_image_no_tint = (np.int32(255) - filter_subtract(denoised_image, tint_background)).astype(np.uint8)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, denoised_image_no_tint)

        logger.info("Converting image to grayscale")
        step += 1
        grayscale_image = cv2.cvtColor(denoised_image_no_tint, cv2.COLOR_BGR2GRAY)
        if gamma_sep != 1.0:
            grayscale_image_gamma_corr = (cv2.pow(grayscale_image.astype(np.double) / 255.0, gamma_sep) * 255.0).astype(np.uint8)
        else:
            grayscale_image_gamma_corr = grayscale_image
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, grayscale_image_gamma_corr)
        
        logger.info("Applying Otsu's threshold")
        step += 1
        _, otsu_mask = cv2.threshold(grayscale_image_gamma_corr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, otsu_mask)
        
        logger.info("Eroding the mask")
        step += 1
        eroded_mask = cv2.erode(otsu_mask, kernel, erode_iterations)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, eroded_mask)

        step -= 100

    logger.info("Processing the foreground")
    step += 1
    if denoise_fg_h == -1 and denoise_fg_h_for_color == -1:
        if onestep:
            logger.debug("Removing the tint from the denoised image")
            denoised_image_fg_no_tint = (np.int32(255) - filter_subtract(denoised_image, tint_background)).astype(np.uint8)
        else:
            denoised_image_fg_no_tint = denoised_image_no_tint
    elif denoise_fg_h == 0 and denoise_fg_h_for_color == 0:
        if onestep:
            logger.debug("Removing the tint from the original image")
            denoised_image_fg_no_tint = (np.int32(255) - filter_subtract(original_image, tint_background)).astype(np.uint8)
        else:
            denoised_image_fg_no_tint = image_no_tint
    else:
        if onestep:
            logger.debug("Denoising the image")
            denoised_image_fg = cv2.fastNlMeansDenoisingColored(original_image, None, denoise_fg_h, denoise_fg_h_for_color, denoise_fg_template_window_size, denoise_fg_search_window_size)
            logger.debug("Removing the tint")
            denoised_image_fg_no_tint = (np.int32(255) - filter_subtract(denoised_image_fg, tint_background)).astype(np.uint8)
        else:
            logger.debug("Denoising the image")
            denoised_image_fg_no_tint = cv2.fastNlMeansDenoisingColored(image_no_tint, None, denoise_fg_h, denoise_fg_h_for_color, denoise_fg_template_window_size, denoise_fg_search_window_size)

    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, denoised_image_fg_no_tint)

    if separate:
        logger.info("Saving the foreground only")
        image_foreground = cv2.cvtColor(denoised_image_fg_no_tint, cv2.COLOR_BGR2BGRA)
        image_foreground[:,:,3] = 255-eroded_mask
        cv2.imwrite(f"{output_path_noext}_foreground{output_ext}", image_foreground)

    logger.info("Processing the background")
    step += 1
    logger.debug("Removing the tint from the inpainted image")
    if onestep:
        denoised_inpainted_background = (np.int32(255) - filter_subtract(inpainted_background, tint_background)).astype(np.uint8)
    if denoise_bg_h != -1 and denoise_bg_h_for_color != -1:
        if denoise_bg_h == denoise_fg_h and denoise_bg_h_for_color == denoise_fg_h_for_color and denoise_bg_search_window_size == denoise_fg_search_window_size and denoise_bg_template_window_size == denoise_fg_template_window_size:
            denoised_image_bg_no_tint = denoised_image_fg_no_tint
        else:
            if onestep:
                if denoise_bg_h == 0 and denoise_bg_h_for_color == 0:
                    denoised_image_bg = original_image
                elif denoise_bg_h == -1 and denoise_bg_h_for_color == -1:
                    logger.error("Cannot be here!")
                else:
                    logger.debug("Denoising the image")
                    denoised_image_bg = cv2.fastNlMeansDenoisingColored(original_image, None, denoise_bg_h, denoise_bg_h_for_color, denoise_bg_template_window_size, denoise_bg_search_window_size)
                logger.debug("Removing the tint")
                denoised_image_bg_no_tint = (np.int32(255) - filter_subtract(denoised_image_bg, tint_background)).astype(np.uint8)
                if save_intermediate:
                    save_image_step(output_path_noext, step, output_ext, denoised_image_bg_no_tint)
            else:
                if denoise_bg_h == 0 and denoise_bg_h_for_color == 0:
                    denoised_image_bg_no_tint = image_no_tint
                else:
                    logger.debug("Denoising the image")
                    denoised_image_bg_no_tint = cv2.fastNlMeansDenoisingColored(image_no_tint, None, denoise_bg_h, denoise_bg_h_for_color, denoise_bg_template_window_size, denoise_bg_search_window_size)
        if onestep:
            logger.info("Substituting the inpainted background")
            denoised_inpainted_background = combine_with_mask(denoised_image_bg_no_tint, denoised_inpainted_background, eroded_mask)
        else:
            if inpaint_bg:
                logger.info("Inpainting the background")
                denoised_inpainted_background = cv2.inpaint(denoised_image_bg_no_tint, cv2.bitwise_not(eroded_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
            else:
                denoised_inpainted_background = denoised_image_bg_no_tint
    else:  # denoise_bg_h == -1 and denoise_bg_h_for_color == -1:
        if not onestep:
            if inpaint_bg:
                logger.info("Inpainting the background")
                denoised_inpainted_background = cv2.inpaint(denoised_image_no_tint, cv2.bitwise_not(eroded_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
            else:
                denoised_inpainted_background = denoised_image_no_tint

    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, denoised_inpainted_background)
    
    if separate:
        logger.info("Saving the background only")
        cv2.imwrite(f"{output_path_noext}_background{output_ext}", denoised_inpainted_background)
    
    logger.info("Combining final images")
    final_output_image = combine_with_mask(denoised_inpainted_background, denoised_image_fg_no_tint, eroded_mask)
    
    logger.info("Saving the final output image")
    cv2.imwrite(output_path, final_output_image)
    logger.info(f"Final output saved to {output_path}")


if __name__ == '__main__':
    main()
