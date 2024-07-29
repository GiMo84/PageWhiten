# PageWhiten

> A tool to remove background color from scanned documents, ideal for yellowed pages or recycled paper.

This project provides a command-line tool to remove the background tint and enhance the foreground of images using various image processing techniques. The tool is implemented in Python and utilizes the OpenCV library for image processing tasks.

## Features

* Background Tint Removal: Removes the background tint from images.
* Background and Foreground Enhancement: Denoises (with separate settings) the background and foreground.
* Separate Foreground and Background saving: Can save foreground and background images separately (e.g. to benefit from different compression settings, as implemented in DjVu).
* Inpainting: Supports inpainting of the background.

## Installation

Ensure you have Python 3.x installed. You can install the required packages using pip:

```sh
pip install numpy opencv-python click click-logging
```

## Usage

The tool shall be executed from the command line. Below are some examples of how to use the tool.

### Basic Usage

```sh
python PageWhiten.py input_image.jpg -o output_image.jpg
```

### Multiple Input Files

```sh
python PageWhiten.py input1.jpg input2.jpg -o output_directory
```

### Troubleshooting

By saving intermediate images, the processing parameters can be tuned more easily.

```sh
python PageWhiten.py input_image.jpg -o output_image.jpg --save-intermediate
```