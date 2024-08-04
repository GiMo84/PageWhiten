import os
import subprocess
import tempfile
import cv2
import numpy as np
import click
import click_logging
import logging
import inspect
from functools import wraps
from collections import Counter
import concurrent.futures

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

def darkest_rgb(image):
    return np.min(image, axis=2)
def lightest_rgb(image):
    return np.max(image, axis=2)

def remove_lines(mask, line_direction='hv', angular_threshold=10, line_thickness=5, rho=1, threshold=100, min_line_length_pct=50, max_line_gap=10):

    def find_common_spacing(lines, orientation='h'):
        # Function to find common spacing
        # TODO improve me... works really badly

        nonlocal line_thickness
        nonlocal mask

        lines_arr = np.array(lines)
        
        x0 = 0
        xmax = mask.shape[1]
        y0 = 0
        ymax = mask.shape[0]

        if len(lines) < 2:
            return []
        
        if orientation == 'h':
            x1s = np.ones(len(lines)) * x0
            x2s = np.ones(len(lines)) * x0
            x1orig = lines_arr[:,0]
            y1orig = lines_arr[:,1]
            x2orig = lines_arr[:,2]
            y2orig = lines_arr[:,3]
            #x1orig, y1orig, x2orig, y2orig = 0
            ms = (y2orig - y1orig) / (x2orig - x1orig)
            qs = ((y1orig - ms * x1orig) + (y2orig - ms * x2orig))/2
            y1s = ms * x1orig + qs
            y2s = ms * x2orig + qs

            positions = (y1s + y2s)/2
            #y1_ext = 
            #positions = np.array([(y1 + y2) / 2 for _, y1, _, y2 in lines])
        else:
            positions = np.array([(x1 + x2) / 2 for x1, _, x2, _ in lines])
        
        # Calculate all pairwise spacings
        spacings = np.abs(positions[:, None] - positions)
        spacings = spacings[np.triu_indices(spacings.shape[0], k=1)]
        
        # Find the most common spacing
        #common_spacing = Counter(spacings).most_common(1)[0][1]
        common_spacing = [sp for sp, _ in Counter(spacings).most_common() if sp >= line_thickness*2][0]
        logger.debug(f"Most common spacing between horizontal lines: {common_spacing} px")
        
        ## Identify lines that follow the common spacing pattern
        #masked_lines = []
        #for position in positions:
        #    close_lines = [line for line in lines if any(np.abs(position - pos) < common_spacing / 2 for pos in positions)]
        #    masked_lines.extend(close_lines)
        #
        #return np.unique(masked_lines)

        # Identify lines that follow the common spacing pattern
        unique_lines = set()
        for position, line in zip(positions, lines):
            if any(np.abs(position - pos) < common_spacing / 2 for pos in positions):
                unique_lines.add(line)
        
        return list(unique_lines), common_spacing
    
    min_line_length = min(mask.shape[0], mask.shape[1]) * min_line_length_pct / 100
    lines = cv2.HoughLinesP(cv2.bitwise_not(mask), rho, 0.1*np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Filter lines by angle and length
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                if length < min_line_length:
                    continue
                
                if abs(angle) < angular_threshold or abs(angle) > (180-angular_threshold):  # Near-horizontal lines
                    horizontal_lines.append((x1, y1, x2, y2))
                elif abs(angle) > (90-angular_threshold) and abs(angle) < (90+angular_threshold):  # Near-vertical lines
                    vertical_lines.append((x1, y1, x2, y2))

    if line_direction.find('h') >= 0:
        _, horizontal_lines_spacing = find_common_spacing(horizontal_lines)
        logger.debug(f"Detected {len(horizontal_lines)} long and near-horizontal lines in the image.")
    if line_direction.find('v') >= 0:
        _, vertical_lines_spacing = find_common_spacing(vertical_lines)
        logger.debug(f"Detected {len(vertical_lines)} long and near-vertical lines in the image.")

    # Create a mask for the detected lines, extending them through the image edges
    line_mask = np.zeros_like(mask)
    if line_direction.find('h') >= 0:
        x0 = 0
        xmax = mask.shape[1]
        m_avg = np.mean([(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in horizontal_lines])
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            #cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_thickness)
            #m = (y2 - y1) / (x2 - x1)
            q = ((y1 - m_avg * x1) + (y2 - m_avg * x2))/2
            cv2.line(line_mask, (x0, round(m_avg*x0+q)), (xmax, round(m_avg*xmax+q)), 255, line_thickness)
    if line_direction.find('v') >= 0:
        y0 = 0
        ymax = mask.shape[0]
        m_avg = np.mean([(x2 - x1) / (y2 - y1) for x1, y1, x2, y2 in horizontal_lines])
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            #cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_thickness)
            #m = (x2 - x1) / (y2 - y1)
            q = ((x1 - m_avg * y1) + (x2 - m_avg * y2))/2
            cv2.line(line_mask, (round(m_avg*y0+q), y0), (round(m_avg*ymax+q), ymax), 255, line_thickness)

    return line_mask

class CommandError(Exception):
    """
    Exception raised when a command fails to execute correctly.

    Attributes:
        message (str): The error message.
        stderr (str): The standard error output of the command.
    """

    def __init__(self, message, stderr):
        super().__init__(message)
        self.stderr = stderr

def run_command(command, input_data=None):
    """
    Executes a command and captures its output.

    Args:
        command (list): The command to execute as a list of strings.
        input_data (bytes, optional): The input data to pass to the command's standard input. Defaults to None.

    Returns:
        tuple: A tuple containing the command's standard output, standard error, and the process object.

    Raises:
        CommandError: If the command fails to execute correctly.

    """
    try:
        logger.debug(f"Running command: {command}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input_data)
        returncode = process.returncode
        logger.debug(f"{os.path.basename(command[0])} [{str(returncode)}]: {stderr.decode()}")
        if len(stdout) > 0:
            logger.debug(f"{os.path.basename(command[0])} returned {len(stdout)} bytes.")
        if returncode != 0:
            raise CommandError("Command {} failed with return code {}: {}".format(command, returncode), stderr.decode())
        return stdout, stderr, process
    except Exception as e:
        raise CommandError("Error running command {}: {}".format(command, str(e)), "")

@click.group()
def cli():
    pass

@cli.command()
@click.argument("inputfiles", type=click.Path(exists=True, file_okay=True), nargs=-1, required=True)
@click.option("--outputfile", "-o", type=click.Path(writable=True), required=True, help="Output file.")
@click.option("--resolution", "-r", default=300, type=click.INT, help="Resolution of the input images (dpi).")
@click.option("--cr-mask", "-m", default=1, type=click.INT,help="Compression ratio for bitonal mask.")
@click.option("--cr-bg", "-b", default="63+13+11", type=click.STRING, help="Compression quality for the background IW44 layer.")
@click.option("--cr-fg", "-f", default=70, type=click.INT, help="Compression quality for the IW44 layer.")
@click.option("--temp-dir", default=None, type=click.Path(file_okay=False, dir_okay=True, writable=True), envvar='TEMPDIR', help="Temporary directory to store intermediate files.")
@click.option("--keep-temp", default=False, is_flag=True, help="Keep temporary files.")
@click.option("--threads", "-t", default=1, type=click.INT, help="Number of threads to use for image processing.")
@click_logging.simple_verbosity_option(logger)
def make_djvu(inputfiles, outputfile, resolution, cr_mask, cr_bg, cr_fg, temp_dir, keep_temp, threads):
    if os.path.exists(outputfile):
        click.confirm(f"Output file {outputfile} already exists. Do you want to continue?", abort=True)
    
    temp_dir_obj = None
    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="djvu_temp_")
        temp_dir = temp_dir_obj.name
        logger.info(f"Using temporary folder: {temp_dir}")
        pages = []

    def process_image(inputfile):
        """
        Process an image file (img.ext) with associated foreground (img_mask_fg.ext), background (img_mask_bg.ext) and separation (img_mask.ext) masks and generates a DjVu file for the page.

        Args:
            inputfile (str): The name of the image file to process.

        Returns:
            tuple: A tuple containing the original filename and the path of the generated DjVu file.
        """
        nonlocal temp_dir
        basedir = os.path.split(inputfile)[0]
        filename = os.path.split(inputfile)[1]
        page_name = os.path.splitext(filename)[0]
        page_ext = os.path.splitext(filename)[1]
        outputfile_tmp = os.path.join(temp_dir, f"{page_name}.djvu")

        if page_ext.lower() != '.tif' and page_ext.lower() != '.tiff':
            logger.error(f"The utility only supports TIFF files.")
            raise Exception(f"The utility only supports TIFF files.") 

        inputfile_mask_fg = os.path.join(basedir, f"{page_name}_mask_fg{page_ext}")
        inputfile_mask_bg = os.path.join(basedir, f"{page_name}_mask_bg{page_ext}")
        inputfile_mask = os.path.join(basedir, f"{page_name}_mask{page_ext}")

        if all([os.path.exists(file) for file in [inputfile, inputfile_mask_fg, inputfile_mask_bg, inputfile_mask]]):
            # Process foreground and background files
            logger.debug(f"Processing page {page_name} with foreground mask {os.path.split(inputfile_mask_fg)[1]}, background mask {os.path.split(inputfile_mask_bg)[1]} and separation mask {os.path.split(inputfile_mask)[1]}.")

            with tempfile.TemporaryDirectory(prefix="PageWhiten_temp_", dir=temp_dir) as temp_page_dir:
                def convert_to_pbm(inputfile, outputfile):
                    out, _, _ = run_command(["tifftopnm", inputfile])
                    with open(outputfile, "wb") as f:
                        f.write(out)
                def extract_iw44(inputfile, outputfile):
                    run_command(["djvuextract", inputfile, "BG44=" + outputfile])
                
                def convert_to_djvu_c44(inputfile, outputfile, dpi, mask, slice):
                    run_command(["c44", inputfile, "-dpi", str(dpi), "-mask", mask, "-slice", str(slice), outputfile])
                
                def convert_to_djvu_cjb2(inputfile, outputfile, dpi, losslevel):
                    run_command(["cjb2", inputfile, "-dpi", str(dpi), "-losslevel", str(losslevel), outputfile])
                
                mask_PBM = os.path.join(temp_page_dir, 'mask.pbm')
                bg_PBM = os.path.join(temp_page_dir, 'bg.pbm')
                fg_PBM = os.path.join(temp_page_dir, 'fg.pbm')
                img_PPM = os.path.join(temp_page_dir, 'img.ppm')
                mask_DJVU = os.path.join(temp_page_dir, 'mask.djvu')
                bg_DJVU = os.path.join(temp_page_dir, 'bg.djvu')
                fg_DJVU = os.path.join(temp_page_dir, 'fg.djvu')
                bg_IW44 = os.path.join(temp_page_dir, 'bg.iw44')
                fg_IW44 = os.path.join(temp_page_dir, 'fg.iw44')

                # Convert input files to PBM format
                for inputfile_conv, outputfile_conv in [(inputfile, img_PPM), (inputfile_mask_fg, fg_PBM), (inputfile_mask_bg, bg_PBM), (inputfile_mask, mask_PBM)]:
                    convert_to_pbm(inputfile_conv, outputfile_conv)

                # Convert mask to JBIG2 DJVU
                convert_to_djvu_cjb2(mask_PBM, mask_DJVU, resolution, cr_mask)

                # Convert image - masked with background mask - to C44 DJVU
                convert_to_djvu_c44(img_PPM, bg_DJVU, resolution, bg_PBM, cr_bg)
                extract_iw44(bg_DJVU, bg_IW44)

                # Convert image - masked with foreground mask - to C44 DJVU
                convert_to_djvu_c44(img_PPM, fg_DJVU, resolution, fg_PBM, cr_fg)
                extract_iw44(fg_DJVU, fg_IW44)

                # Combine background and foreground
                run_command(["djvumake", outputfile_tmp, f"INFO=,,{resolution}", f"Sjbz={mask_DJVU}", f"BG44={bg_IW44}", f"FG44={fg_IW44}"])

            return inputfile, outputfile_tmp
        else:
            logger.error("Not all files exist. Skipping page.")
            return None


    # Process images from input directory3
    if threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_image, filename) for filename in inputfiles]
            if logger.level >= logging.INFO:
                # Enable progress bar
                futures_iterator = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing pages")
            else:
                # Disable progress bar: debug output would ruin it
                futures_iterator = concurrent.futures.as_completed(futures)

            for future in futures_iterator:
                if future.result() is not None:
                    filename, page = future.result()
                    if page:
                        pages.append((filename, page))
    else:
        for filename_in in inputfiles:
            filename, page = process_image(filename_in)
            pages.append((filename, page))

    # Restore pages order
    pages.sort(key=lambda x: sorted(inputfiles).index(x[0]))
    pages = [page for _, page in pages]

    # Assemble all pages into a single DjVu file
    if pages:
        if os.path.exists(outputfile):
            logger.debug("Removing old output file.")
            os.remove(outputfile)
        run_command(["djvm", "-c", outputfile] + pages)
        logger.info(f"DjVu file '{outputfile}' created successfully!")
        if not keep_temp and temp_dir_obj is not None:
            logger.debug("Cleaning up temporary folder.")
            temp_dir_obj.cleanup()
    else:
        logger.error("No pages found to assemble DjVu file.")

@cli.command()
@click.argument("input", type=click.Path(exists=True, file_okay=True), nargs=-1, required=True)
@click.option("--output", "-o", "output_path", default='.', type=click.Path(writable=True), help="Output file or path.")
@click.option("--outfmt", default='', type=click.STRING, help="Output file format (extension).")
@click.option("--gamma-tint-mask", "-gt", default=1.0, type=click.FLOAT, help="Gamma correction for the tint mask.")
@click.option("--gamma-sep", "-gs", default=1.0, type=click.FLOAT, help="Gamma correction for the foreground/background separation mask (not compatible with --onestep).")
@click.option('--denoise-tint-mask', nargs=4, default=(10, 10, 7, 21), type=click.Tuple([click.INT, click.INT, click.INT, click.INT]), help='Denoising parameters for the tint mask.')
@click.option('--denoise-bg', nargs=4, default=(-1, -1, -1, -1), type=click.Tuple([click.INT, click.INT, click.INT, click.INT]), help='Denoising parameters for the background. 0 disables denoising, -1 uses tint-mask denoising parameters.')
@click.option('--denoise-fg', nargs=4, default=(0, 0, 0, 0), type=click.Tuple([click.INT, click.INT, click.INT, click.INT]), help='Denoising parameters for the foreground. 0 disables denoising, -1 uses tint-mask denoising parameters.')
@click.option('--inpaint-bg', is_flag=True, help='Inpaint the background.')
@click.option('--remove-lines-dir', default='', type=click.Choice(['', 'h', 'v', 'hv']), help='Find and remove horizontal (h) and/or vertical (v) lines.')
@click.option('--lines-thickness', default=5, type=click.INT, help='Thickness of the lines.')
@click.option('--lines-angle-deviation', default=10.0, type=click.FLOAT, help='Maximum deviation from the horizontal/vertical of the lines to remove.')
@click.option('--lines-min-length', default=10.0, type=click.FLOAT, help='Minimum size, in percentage of image min(height, widht), of the lines.')
@click.option('--kernel-size', default=5, type=click.INT, help='Kernel size for morphological operations.')
@click.option('--erode-iterations', default=1, type=click.INT, help='Number of iterations for the mask erosion operation.')
@click.option('--dilate-erode-iterations', default=100, type=click.INT, help='Number of iterations for the blurred background mask dilation and erosion operation.')
@click.option('--save-intermediate', '-i', is_flag=True, help='Flag to save intermediate images.')
@click.option('--separate', '-s', is_flag=True, help='Flag to save separately foreground and background images.')
@click.option('--save-mask', '-m', is_flag=True, help='Flag to save a bitonal foreground (black)/background (white) mask.')
@click.option('--onestep', is_flag=True, help='Flag to use one-step process (forces --inpaint-bg).')
@click_logging.simple_verbosity_option(logger)
def clean(input, output_path, outfmt, gamma_tint_mask, gamma_sep, denoise_tint_mask, denoise_bg, denoise_fg, inpaint_bg, remove_lines_dir, lines_thickness, lines_angle_deviation, lines_min_length, kernel_size, erode_iterations, dilate_erode_iterations, save_intermediate, separate, save_mask, onestep):
    """
    Removes the background tint and enhances the image.
    """

    if len(outfmt) > 0:
        if outfmt[0] != '.':
            outfmt = '.' + outfmt

    if len(input) == 1:
        if os.path.isdir(output_path):
            outfile = os.path.split(input[0])[1]
            outfname = os.path.splitext(outfile)[0]
            if len(outfmt) == 0:
                outfext = os.path.splitext(fname)[1]
            else:
                outfext = outfmt
            output_file = os.path.join(output_path, outfname + '_out' + outfext)
        else:
            output_file = output_path
    else:
        if not(os.path.isdir(output_path)):
            logger.error("Output must be a path if multiple input files are given.")
            raise NotADirectoryError
    for input_file in input:
        if len(input) > 1:
            fname = os.path.split(input_file)[1]
            if len(outfmt) == 0:
                outfext = os.path.splitext(fname)[1]
            else:
                outfext = outfmt
            output_file = os.path.join(output_path, os.path.splitext(fname)[0] + '_out' + outfext)
        click.echo(f"Processing {input_file} -> {output_file}")
        clean_image(input_file, output_file, gamma_tint_mask, gamma_sep, denoise_tint_mask, denoise_bg, denoise_fg, inpaint_bg, remove_lines_dir, lines_thickness, lines_angle_deviation, lines_min_length, kernel_size, erode_iterations, dilate_erode_iterations, save_intermediate, separate, save_mask, onestep)

def clean_image(image_path, output_path, gamma_tint_mask, gamma_sep, denoise_tint_mask, denoise_bg, denoise_fg, inpaint_bg, remove_lines_dir, lines_thickness, lines_angle_deviation, lines_min_length, kernel_size, erode_iterations, dilate_erode_iterations, save_intermediate, separate, save_mask, onestep):
    """
    Process an image by performing a series of operations to remove the background tint and enhance the foreground.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to save the processed image.
        gamma_tint_mask (float): Gamma correction for the tint mask.
        gamma_sep (float): Gamma correction for the foreground/background separation mask (not compatible with --onestep).
        denoise_tint_mask (tuple): Denoising parameters for the tint mask.
        denoise_bg (tuple): Denoising parameters for the background. 0 disables denoising, -1 uses tint_mask denoising parameters.
        denoise_fg (tuple): Denoising parameters for the foreground. 0 disables denoising, -1 uses tint_mask denoising parameters.
        inpaint_bg (bool): Inpaint the background.
        remove_lines_dir (str): Remove horizontal (h) and/or vertical (v) lines from the image.
        lines_thickness (int): Thickness of the lines to remove.
        lines_angle_deviation (float): Maximum deviation from the horizontal/vertical of the lines to remove.
        lines_min_length (float): Minimum size, in percentage of image min(height, widht), of the lines.
        kernel_size (int): Kernel size for morphological operations.
        erode_iterations (int): Number of iterations for the mask erosion operation.
        dilate_erode_iterations (int): Number of iterations for the blurred background mask dilation and erosion operation.
        save_intermediate (bool): Flag to save intermediate images.
        separate (bool): Flag to save separately foreground and background images.
        save_mask (bool): Flag to save a bitonal foreground (black)/background (white) mask.
        onestep (bool): Flag to use one_step process (forces --inpaint-bg).
    """
    output_path_noext = os.path.splitext(output_path)[0]
    output_ext = os.path.splitext(output_path)[1]

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
    #grayscale_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    grayscale_image = darkest_rgb(denoised_image)
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

    #logger.info("Enhancing the mask using color info")
    #step += 1
    #color_mask = mask_from_color(denoised_image)
    #if save_intermediate:
    #    save_image_step(output_path_noext, step, output_ext, color_mask)
    #
    #logger.info("Combining masks")
    #step += 1
    #combined_mask = cv2.bitwise_and(otsu_mask, color_mask)
    #if save_intermediate:
    #    save_image_step(output_path_noext, step, output_ext, combined_mask)
    combined_mask = otsu_mask

    step += 1
    if len(remove_lines_dir) > 0:
        logger.info("Removing lines")
        lines_mask = remove_lines(combined_mask, line_direction=remove_lines_dir, angular_threshold=lines_angle_deviation, line_thickness=lines_thickness, min_line_length_pct=lines_min_length)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, lines_mask)
        
        # Inpaint over the lines
        original_image_no_lines = cv2.inpaint(original_image, lines_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        denoised_image_no_lines = cv2.inpaint(denoised_image, lines_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        combined_mask_no_lines = cv2.threshold(cv2.inpaint(combined_mask, lines_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA), 127, 255, cv2.THRESH_BINARY)[1]
        combined_mask = combined_mask_no_lines

        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, original_image_no_lines)
            save_image_step(output_path_noext, step, output_ext, denoised_image_no_lines)
            save_image_step(output_path_noext, step, output_ext, combined_mask_no_lines)
    else:
        denoised_image_no_lines = denoised_image
        original_image_no_lines = original_image

    logger.info("Eroding the mask")
    step += 1
    eroded_mask = cv2.erode(combined_mask, kernel, erode_iterations)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, eroded_mask)

    logger.info("Inpainting the background")
    step += 1
    inpainted_background = cv2.inpaint(denoised_image_no_lines, cv2.bitwise_not(eroded_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
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
            image_no_tint = (np.int16(255) - filter_subtract(original_image_no_lines, tint_background)).astype(np.uint8)
            if save_intermediate:
                save_image_step(output_path_noext, step, output_ext, image_no_tint)

        logger.info("Removing the tint from the denoised image")
        step += 1
        denoised_image_no_lines_no_tint = (np.int16(255) - filter_subtract(denoised_image_no_lines, tint_background)).astype(np.uint8)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, denoised_image_no_lines_no_tint)

        logger.info("Converting image to grayscale")
        step += 1
        #grayscale_image = cv2.cvtColor(denoised_image_no_lines_no_tint, cv2.COLOR_BGR2GRAY)
        grayscale_image = darkest_rgb(denoised_image_no_lines_no_tint)
        if gamma_sep != 1.0:
            grayscale_image_gamma_corr = (cv2.pow(grayscale_image.astype(np.double) / 255.0, gamma_sep) * 255.0).astype(np.uint8)
        else:
            grayscale_image_gamma_corr = grayscale_image
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, grayscale_image_gamma_corr)
        
        logger.info("Applying Otsu's threshold")
        step += 1
        _, fg_bg_mask = cv2.threshold(grayscale_image_gamma_corr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if save_intermediate:
            save_image_step(output_path_noext, step, output_ext, fg_bg_mask)

        step -= 100
    else:
        fg_bg_mask = combined_mask
    
    logger.info("Eroding the mask")
    step += 1
    eroded_mask = cv2.erode(fg_bg_mask, kernel, erode_iterations)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, eroded_mask)

    logger.info("Dilating the mask")
    step += 1
    dilated_mask = cv2.dilate(fg_bg_mask, kernel, erode_iterations)
    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, dilated_mask)

    logger.info("Processing the foreground")
    step += 1
    if denoise_fg_h == -1 and denoise_fg_h_for_color == -1:
        if onestep:
            logger.debug("Removing the tint from the denoised image")
            denoised_image_no_lines_fg_no_tint = (np.int32(255) - filter_subtract(denoised_image_no_lines, tint_background)).astype(np.uint8)
        else:
            denoised_image_no_lines_fg_no_tint = denoised_image_no_lines_no_tint
    elif denoise_fg_h == 0 and denoise_fg_h_for_color == 0:
        if onestep:
            logger.debug("Removing the tint from the original image")
            denoised_image_no_lines_fg_no_tint = (np.int32(255) - filter_subtract(original_image_no_lines, tint_background)).astype(np.uint8)
        else:
            denoised_image_no_lines_fg_no_tint = image_no_tint
    else:
        if onestep:
            logger.debug("Denoising the image")
            denoised_image_no_lines_fg = cv2.fastNlMeansDenoisingColored(original_image_no_lines, None, denoise_fg_h, denoise_fg_h_for_color, denoise_fg_template_window_size, denoise_fg_search_window_size)
            logger.debug("Removing the tint")
            denoised_image_no_lines_fg_no_tint = (np.int32(255) - filter_subtract(denoised_image_no_lines_fg, tint_background)).astype(np.uint8)
        else:
            logger.debug("Denoising the image")
            denoised_image_no_lines_fg_no_tint = cv2.fastNlMeansDenoisingColored(image_no_tint, None, denoise_fg_h, denoise_fg_h_for_color, denoise_fg_template_window_size, denoise_fg_search_window_size)

    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, denoised_image_no_lines_fg_no_tint)

    if separate:
        logger.info("Saving the foreground only")
        image_foreground = cv2.cvtColor(denoised_image_no_lines_fg_no_tint, cv2.COLOR_BGR2BGRA)
        image_foreground[:,:,3] = 255-fg_bg_mask
        cv2.imwrite(f"{output_path_noext}_foreground{output_ext}", image_foreground)

    logger.info("Processing the background")
    step += 1
    logger.debug("Removing the tint from the inpainted image")
    if onestep:
        denoised_inpainted_background = (np.int32(255) - filter_subtract(inpainted_background, tint_background)).astype(np.uint8)
    if denoise_bg_h != -1 and denoise_bg_h_for_color != -1:
        if denoise_bg_h == denoise_fg_h and denoise_bg_h_for_color == denoise_fg_h_for_color and denoise_bg_search_window_size == denoise_fg_search_window_size and denoise_bg_template_window_size == denoise_fg_template_window_size:
            denoised_image_no_lines_bg_no_tint = denoised_image_no_lines_fg_no_tint
        else:
            if onestep:
                if denoise_bg_h == 0 and denoise_bg_h_for_color == 0:
                    denoised_image_no_lines_bg = original_image_no_lines
                elif denoise_bg_h == -1 and denoise_bg_h_for_color == -1:
                    logger.error("Cannot be here!")
                else:
                    logger.debug("Denoising the image")
                    denoised_image_no_lines_bg = cv2.fastNlMeansDenoisingColored(original_image_no_lines, None, denoise_bg_h, denoise_bg_h_for_color, denoise_bg_template_window_size, denoise_bg_search_window_size)
                logger.debug("Removing the tint")
                denoised_image_no_lines_bg_no_tint = (np.int32(255) - filter_subtract(denoised_image_no_lines_bg, tint_background)).astype(np.uint8)
                if save_intermediate:
                    save_image_step(output_path_noext, step, output_ext, denoised_image_no_lines_bg_no_tint)
            else:
                if denoise_bg_h == 0 and denoise_bg_h_for_color == 0:
                    denoised_image_no_lines_bg_no_tint = image_no_tint
                else:
                    logger.debug("Denoising the image")
                    denoised_image_no_lines_bg_no_tint = cv2.fastNlMeansDenoisingColored(image_no_tint, None, denoise_bg_h, denoise_bg_h_for_color, denoise_bg_template_window_size, denoise_bg_search_window_size)
        if onestep:
            logger.info("Substituting the inpainted background")
            denoised_inpainted_background = combine_with_mask(denoised_image_no_lines_bg_no_tint, denoised_inpainted_background, fg_bg_mask)
        else:
            if inpaint_bg:
                logger.info("Inpainting the background")
                denoised_inpainted_background = cv2.inpaint(denoised_image_no_lines_bg_no_tint, cv2.bitwise_not(fg_bg_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
            else:
                denoised_inpainted_background = denoised_image_no_lines_bg_no_tint
    else:  # denoise_bg_h == -1 and denoise_bg_h_for_color == -1:
        if not onestep:
            if inpaint_bg:
                logger.info("Inpainting the background")
                denoised_inpainted_background = cv2.inpaint(denoised_image_no_lines_no_tint, cv2.bitwise_not(fg_bg_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
            else:
                denoised_inpainted_background = denoised_image_no_lines_no_tint

    if save_intermediate:
        save_image_step(output_path_noext, step, output_ext, denoised_inpainted_background)
    
    if separate:
        logger.info("Saving the background only")
        cv2.imwrite(f"{output_path_noext}_background{output_ext}", denoised_inpainted_background)
    
    logger.info("Combining final images")
    final_output_image = combine_with_mask(denoised_inpainted_background, denoised_image_no_lines_fg_no_tint, fg_bg_mask)

    if False:
        inpainted_foreground = cv2.inpaint(denoised_image_no_lines_fg_no_tint, (fg_bg_mask), inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(f"{output_path_noext}_fg{output_ext}", inpainted_foreground)
        cv2.imwrite(f"{output_path_noext}_bg{output_ext}", denoised_inpainted_background)

    if save_mask:
        logger.info("Saving the background/foreground mask")
        cv2.imwrite(f"{output_path_noext}_mask{output_ext}", fg_bg_mask)
        #cv2.imwrite(f"{output_path_noext}_mask_inv{output_ext}", cv2.bitwise_not(fg_bg_mask))
        if separate:
            logger.info("Saving the foreground mask")
            cv2.imwrite(f"{output_path_noext}_mask_fg{output_ext}", cv2.bitwise_not(dilated_mask))
            logger.info("Saving the background mask")
            cv2.imwrite(f"{output_path_noext}_mask_bg{output_ext}", (eroded_mask))

    logger.info("Saving the final output image")
    cv2.imwrite(output_path, final_output_image)
    logger.info(f"Final output saved to {output_path}")


if __name__ == '__main__':
    cli()
