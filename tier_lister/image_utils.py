"""Stuff to generate images"""

from collections.abc import Iterable
from typing import Any

import numpy
from PIL import Image, ImageColor, ImageFilter, ImageFont
from PIL.ImageDraw import ImageDraw


def generate_nice_background(width: int, height: int, radius: int = 50) -> Image.Image:
	"""Generate some nice pretty rainbow clouds
	Unfortunuately ModeFilter is very slow"""
	rng = numpy.random.default_rng()
	noise = rng.integers(0, (128, 128, 128), (height, width, 3), 'uint8', endpoint=True)
	# Could also have a fourth dim with max=255 and then layer multiple transparent clouds on top of each other hmm maybe
	image = Image.fromarray(noise)
	image = image.filter(ImageFilter.ModeFilter(radius * 2))
	return image.filter(ImageFilter.GaussianBlur(radius))


def pad_image(
	image: Image.Image, width: int, height: int, colour: Any, *, centred: bool = False
) -> Image.Image:
	"""Return expanded version of image with blank space to ensure a certain size.

	Don't forget to call ImageDraw again"""
	new_image = Image.new(image.mode, (width, height), colour)
	if image.palette:
		palette = image.getpalette()
		assert (
			palette
		), 'image.getpalette() should not return None since we have already checked image.palette'
		new_image.putpalette(palette)
	if centred:
		x = (width - image.width) // 2
		y = (height - image.height) // 2
	else:
		x = y = 0
	new_image.paste(image, (x, y))
	return new_image


def draw_box(image: Image.Image, colour: Any = 'black', width: int = 2) -> Image.Image:
	"""Modifies image in-place and returns it with a border around the sides"""
	draw = ImageDraw(image)
	draw.rectangle((0, 0, image.width, image.height), outline=colour, width=width)
	return image


def fit_font(
	font: ImageFont.FreeTypeFont | None,
	width: int | None,
	height: int | None,
	text: str | Iterable[str],
	vertical_padding: int = 10,
	horizontal_padding: int = 10,
) -> tuple[ImageFont.FreeTypeFont, int, int]:
	"""Find the largest font that will fit into a box of a given size, or just height or width to also find how big the other dimension of the box needs to be

	Returns the font, height, or width

	:param font: Font to use, or load the default font if None
	:param width: Max width in pixels or None
	:param height: Max height in pixels or None
	:param text: Text to measure, or iterable of texts to find what fits all of them
	:param vertical_padding: Amount of vertical space in pixels to add to the box
	:param horizontal_padding: Amount of horizontal space in pixels to add to the box
	:raises RuntimeError: if font is None and default font can't be resized
	:raises ValueError: if it can't find a font at all somehow
	"""
	# font_size is points and not pixels, but it'll do as a starting point
	font_size = font.size if font else (100 if height is None else height * 2)

	if isinstance(text, str):
		while True:
			if isinstance(font, ImageFont.FreeTypeFont):
				font = font.font_variant(size=font_size)
			else:
				default_font = ImageFont.load_default(font_size)
				if not isinstance(default_font, ImageFont.FreeTypeFont):
					raise RuntimeError('Uh oh, you need FreeType and Pillow >= 10.1.10')  # noqa: TRY004 #I dunno what would be a better one to raise here
				font = default_font
			if '\n' in text:
				# This is more annoying then, because there is no ImageFont.getmultilinebbox
				# And I don't feel like doing all the line calculations myself to make it work (or do I) (would need to call getbbox for each line and then add the line spacing)
				# This isn't even necessarily a good idea if it wasn't for me never fucking with the spacing parameter of multiline_text when I use it or anything like that
				size = ImageDraw(Image.new('RGBA', (width, height))).multiline_textbbox(
					(0, 0), text, font
				)[2:]
			else:
				size = font.getbbox(text)[2:]
			if font_size == 1 or (
				(height is None or (size[1] + vertical_padding) <= height)
				and (width is None or (size[0] + horizontal_padding) <= width)
			):
				break
			font_size -= 1

		assert font, 'how did my font end up being None :('
		return font, size[0] + horizontal_padding, size[1] + vertical_padding

	out_width = 0
	out_height = 0
	out_font = font
	for t in text:
		t_font, t_width, t_height = fit_font(
			font, width, height, t, vertical_padding, horizontal_padding
		)
		out_width = max(t_width, out_width)
		out_height = max(t_height, out_height)
		out_font = t_font if out_font is None or t_font.size < out_font.size else out_font
	if not out_font:
		raise ValueError(
			'out_font ended up being None, maybe you provided an empty iterable for text'
		)
	return out_font, out_width, out_height


def combine_images_diagonally(first_image: Image.Image, second_image: Image.Image) -> Image.Image:
	"""Return a new image of the size of the first image with one diagonal half being from the first image, and the second half being from the second image"""
	if first_image.size != second_image.size:
		second_image = second_image.resize(first_image.size)
	# numpy.triu/tril won't work nicely on non-square rectangles
	orig_size = first_image.size
	max_dim = max(orig_size)
	square_size = max_dim, max_dim
	a = numpy.array(first_image.resize(square_size))
	b = numpy.array(second_image.resize(square_size))
	# triu/tril works on the last two axes, so we want those to be height and width
	upper_right = numpy.triu(a.swapaxes(0, 2)).swapaxes(0, 2)
	lower_left = numpy.tril(b.swapaxes(0, 2)).swapaxes(0, 2)
	return Image.fromarray(upper_right + lower_left).resize(orig_size)


def draw_centred_textbox(
	draw: ImageDraw,
	background_colour: tuple[float, float, float, float] | str,
	left: int,
	top: int,
	right: int,
	bottom: int,
	text: str,
	font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
):
	"""Draw a box with text in the centre with a 1 pixel border and the specified background colour, selecting white or black text colour as appropriate for readability"""
	if isinstance(background_colour, str):
		colour_as_int = ImageColor.getrgb(background_colour)
		background_colour = tuple(v / 255 for v in colour_as_int)
		assert background_colour, 'Mypy, why would it be a str at this point'
	else:
		colour_as_int = tuple(int(v * 255) for v in background_colour)
	# Am I stupid, or is there actually nothing in standard library or matplotlib that does this
	# Well colorsys.rgb_to_yiv would also potentially work
	luminance = (
		(background_colour[0] * 0.2126)
		+ (background_colour[1] * 0.7152)
		+ (background_colour[2] * 0.0722)
	)
	text_colour = 'white' if luminance <= 0.5 else 'black'
	draw.rectangle((left, top, left, bottom - 1), fill='black')
	draw.rectangle((right, top, right, bottom - 1), fill='black')
	draw.rectangle(
		(left + 1, top, right - 1, bottom - 1), fill=colour_as_int, outline='black', width=1
	)
	draw.text(
		((left + right) // 2, (top + bottom) // 2),
		text,
		anchor='mm',
		fill=text_colour,
		font=font,
		align='center',
	)
