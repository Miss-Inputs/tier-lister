import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Generic, Literal, Protocol, Self, SupportsFloat, TypeVar

import numpy
import pandas
from matplotlib import pyplot
from PIL import Image, ImageFont
from PIL.ImageDraw import ImageDraw

from tier_lister import image_utils
from tier_lister.kmeans_tierer import kmeans_tierer
from tier_lister.tiers import Tiers

if TYPE_CHECKING:
	from matplotlib.colors import Colormap
	from pandas.core.groupby.generic import DataFrameGroupBy

	from tier_lister.tiers import AutoTierer


T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

ItemScore = SupportsFloat | pandas.Timestamp | timedelta
"""A value used for automatic tiering, can be anything that can be converted to a float."""


@dataclass
class TieredItem(Generic[T_co]):
	"""An item that has a score associated with it used for ranking.

	item should not be None, and score should be a higher number for better items.
	"""

	item: T_co
	score: ItemScore


class BaseTierList(ABC, Generic[T]):
	"""Abstract tier list where all the logic is, just add a get_item_image method."""

	def __init__(
		self,
		items: Iterable[TieredItem[T]],
		tiers: 'Sequence[int] | AutoTierer' = kmeans_tierer,
		num_tiers: int | Literal['auto'] = 'auto',
		tier_names: Sequence[str] | Mapping[int, str] | None = None,
		title: str | None = None,
		*,
		append_minmax_to_tier_titles: bool = False,
		score_formatter: str = '',
	) -> None:
		"""
		:param tiers: If a sequence, pre-defined tiers, or a callable that generates tiers automatically from scores
		:param num_tiers: Number of tiers to separate scores into. If "auto", tries to find a number of tiers that balances the size of each tier, but I made up that algorithm myself and I don't strictly speaking know what I'm doing so maybe it doesn't work."""
		self._data = pandas.DataFrame(list(items), columns=['item', 'score'])
		self._data.sort_values('score', ascending=False, inplace=True)
		self._data['rank'] = numpy.arange(1, self._data.index.size + 1)

		if isinstance(tiers, Sequence):
			self._data['tier'] = tiers
			self.num_tiers = self._groupby.ngroups
			centroids = self._groupby['score'].mean().to_dict()
			self.tiers = Tiers(self._data['tier'], centroids)
		else:
			self.tiers = tiers(self._data['score'], num_tiers)
			self.num_tiers = (
				self.tiers.tier_ids.nunique()
			)  # Might not be the same as num_tiers, esp if tiers is 'auto'
			self._data['tier'] = self.tiers.tier_ids

		self.append_minmax_to_tier_titles = append_minmax_to_tier_titles
		self.score_formatter = score_formatter
		if not tier_names:
			tier_names = self.default_tier_names(self.num_tiers)
		if isinstance(tier_names, Sequence):
			tier_names = dict(enumerate(tier_names))
		self.tier_names = tier_names

		self.title = title

	@classmethod
	def from_items(
		cls,
		s: 'Mapping[T, ItemScore] | pandas.Series[float]',
		tiers: 'Sequence[int] | AutoTierer' = kmeans_tierer,
		num_tiers: int | Literal['auto'] = 'auto',
		tier_names: Sequence[str] | Mapping[int, str] | None = None,
		title: str | None = None,
		**kwargs,
	) -> 'Self':
		# Note: T apparently isn't allowed to be covariant here, so that's stupid, means that CustomImageTierList or anything like that has to override this with a more specific type signature
		"""Create a new tier list from scores in a dict or pandas Series.
		If Series, assumes s is indexed by the items to be tiered."""
		return cls(
			itertools.starmap(TieredItem, s.items()), tiers, num_tiers, tier_names, title, **kwargs
		)

	@staticmethod
	def default_tier_names(length: int) -> Mapping[int, str]:
		"""Default tier names named after letters as is usually the case, starting with S at the top"""
		if length == 2:
			return {0: 'Good', 1: 'Bad'}
		if length == 3:
			return {0: 'Good', 1: 'Okay', 2: 'Bad'}
		if length == 6:
			# Just think it looks a bit weird to have it end at E tier
			return dict(enumerate('SABCDF'))
		tier_letters: Sequence[str] = 'SABCDEFGHIJKLMNOPQRSTUVXY'[:length]
		if length > 9:
			# Once you go past H it looks weird, so have the last one be Z
			tier_letters = list(tier_letters)
			tier_letters[-1] = 'Z'
		return dict(enumerate(tier_letters))

	def to_text(self, *, show_scores: bool = False) -> str:
		"""Return this tier list displayed as plain text"""
		text = '\n'.join(
			itertools.chain.from_iterable(
				(
					'=' * 20,
					self.displayed_tier_text(tier_number, group),
					'-' * 10,
					*(
						f'{row.rank}: {row.item}'
						+ (f' ({row.score:{self.score_formatter}})' if show_scores else '')
						for row in group.itertuples()
					),
					'',
				)
				for tier_number, group in self._groupby
			)
		)
		if self.title is not None:
			text = f'{self.title}\n' + text
		return text

	@cached_property
	def _groupby(self) -> 'DataFrameGroupBy[int]':
		return self._data.groupby('tier')  # type: ignore[returnType] #it will return an int grouper, sources: trust me bro

	@cached_property
	def _tier_texts(self) -> Mapping[int, str]:
		"""Based off tier names, not necessarily using all of them if there aren't as many tiers, falling back to a default name if needed"""
		# It is possible, though unlikely, that the groups are not an exact ascending sequence, but tier_names are
		return {
			tier_number: self.tier_names.get(i, f'Tier {tier_number}')
			for i, tier_number in enumerate(self._groupby.groups)
		}  # type: ignore[returnType]

	def displayed_tier_text(self, tier_number: int, group: pandas.DataFrame | None = None):
		"""
		:param tier_number: Tier number
		:param group: If you are already iterating through ._groupby, you can pass each group so you don't have to call get_group"""
		text = self._tier_texts[tier_number]
		if self.append_minmax_to_tier_titles:
			if group is None:
				group = self._groupby.get_group(tier_number)
			min_ = group['score'].min()
			max_ = group['score'].max()
			if numpy.isclose(min_, max_):
				return f'{text} ({min_:{self.score_formatter}})'
			return f'{text} ({min_:{self.score_formatter}} to {max_:{self.score_formatter}})'
		return text

	@abstractmethod
	def get_item_image(self, item: T) -> Image.Image:
		"""Return an image that represents item."""

	@cached_property
	def images(self) -> Mapping[T, Image.Image]:
		return {item: self.get_item_image(item) for item in self._data['item']}

	def to_image(
		self,
		colourmap: 'str | Colormap | None' = None,
		max_images_per_row: int | None = 8,
		*,
		show_scores: bool = False,
		score_height: float = 0.5,
		title_background: tuple[int, int, int, int] | str | None = None,
		background: str
		| tuple[float, float, float, float]
		| Callable[[int, int], Image.Image]
		| Image.Image
		| None = image_utils.generate_nice_background,
	) -> Image.Image:
		"""Render the tier list as an image.

		This doesn't look too great if the images are of uneven size, but that's allowed.
		:param background: A background color (as RGBA, or a colour name accepted by Pillow), a background image, which will be cropped to fit if not the right size, or a callable returning an image given the height/width of the final tier list, or nothing."""
		max_image_width = max(im.width for im in self.images.values())
		max_image_height = max(im.height for im in self.images.values())
		if show_scores:
			scores = {
				item: f'{score:{self.score_formatter}}'
				for item, score in zip(self._data['item'], self._data['score'], strict=True)
			}
			score_font, _, score_height = image_utils.fit_font(
				None,
				max_image_width,
				int(score_height if score_height > 1 else max_image_height * score_height),
				scores.values(),
				vertical_padding=0,
			)
			max_image_height += score_height

		tier_texts = {i: self.displayed_tier_text(i) for i in self._tier_texts}

		if colourmap is None or isinstance(colourmap, str):
			colourmap = pyplot.get_cmap(colourmap)

		font: ImageFont.FreeTypeFont | None = None
		font, textbox_width, _ = image_utils.fit_font(
			font,
			None,
			max_image_height,
			tier_texts.values(),
			horizontal_padding=max_image_height,  # Not a typo, I just think that looks better
		)

		height = self._groupby.ngroups * max_image_height
		if self.title is not None:
			height += max_image_height
		max_tier_size = self._groupby.size().max()
		assert not isinstance(max_tier_size, pandas.Series)
		if not max_images_per_row:
			max_images_per_row = max_tier_size
		width = min(max_tier_size, max_images_per_row) * max_image_width + textbox_width

		has_static_background = isinstance(background, (str, tuple))
		bg_color = background if has_static_background else (0, 0, 0, 0)
		image = Image.new('RGBA', (width, height), bg_color)
		draw = ImageDraw(image)

		next_line_y = 0
		if self.title is not None:
			next_line_y = max_image_height
			title_font = image_utils.fit_font(None, width, max_image_height, self.title)[0]
			image_utils.draw_centred_textbox(
				draw,
				title_background or bg_color,
				0,
				0,
				width,
				max_image_height,
				self.title,
				title_font,
			)

		actual_width = width
		for tier_number, group in self._groupby:
			tier_text = tier_texts[tier_number]

			row_height = max_image_height * (((group.index.size - 1) // max_images_per_row) + 1)

			box_end = next_line_y + row_height
			if box_end > image.height:
				image = image_utils.pad_image(image, image.width, box_end, bg_color)
				draw = ImageDraw(image)
			if textbox_width > image.width:  # This probably doesn't happen
				image = image_utils.pad_image(image, textbox_width, image.height, bg_color)
				draw = ImageDraw(image)

			colour = (
				colourmap(self.tiers.scaled_centroids[tier_number])
				if self.tiers.scaled_centroids
				else 'white'
			)
			# TODO: Default colour for text boxes where the tierer doesn't define centroids should probs be a parameter I guess
			image_utils.draw_centred_textbox(
				draw, colour, 0, next_line_y, textbox_width, box_end, tier_text, font
			)

			next_image_x = textbox_width + 1  # Account for border
			for i, item in enumerate(group['item']):
				item_image = self.images[item]
				
				if show_scores:
					orig_height = item_image.height
					item_image = image_utils.pad_image(
						item_image, item_image.width, orig_height + score_height, bg_color
					)
					image_utils.draw_centred_textbox(
						ImageDraw(item_image),
						'black',
						0,
						orig_height,
						item_image.width,
						item_image.height,
						scores[item],
						score_font,
					)

				image_row, image_col = divmod(i, max_images_per_row)
				if not image_col:
					next_image_x = textbox_width + 1

				image_y = next_line_y + (max_image_height * image_row)
				if next_image_x + item_image.width > image.width:
					image = image_utils.pad_image(
						image, next_image_x + item_image.width, image.height, bg_color
					)
					draw = ImageDraw(image)
				image.paste(item_image, (next_image_x, image_y), item_image)
				next_image_x += item_image.width
				actual_width = max(actual_width, next_image_x)
			next_line_y = box_end

		# Uneven images can result in calculating too much space to the side
		image = image.crop((0, 0, actual_width, next_line_y))

		if background is None or has_static_background:
			return image

		if isinstance(background, Image.Image):
			bg = background.crop((0, 0, image.width, image.height))
		else:
			bg = background(image.width, image.height)
		if bg.mode != 'RGBA':
			bg = bg.convert('RGBA')
		bg.paste(image, mask=image)
		return bg


class TierList(BaseTierList[T]):
	"""Default implementation of TierList that just displays text as images. Doesn't do it nicely, you probably don't want to use this directly.

	If the items are not strings, this will look for a .name property, or use str() to convert them."""

	# Default size of load_default 10, so that sucks and we won't do that
	_default_font = ImageFont.load_default(20)
	# TODO: Let the font be chooseable

	_reg = re.compile(r'\s+?(?=\b\w{5,})')
	"""Space followed by at least 5-letter word, for attempting to nicely break things up into multiple lines"""

	def get_item_image(self, item: T) -> Image.Image:
		text = getattr(item, 'name', str(item))
		if ' ' in text:
			text = TierList._reg.sub('\n', text)
			width, height = ImageDraw(Image.new('1', (1, 1))).multiline_textbbox(
				(0, 0), text, font=TierList._default_font
			)[2:]
		else:
			width, height = TierList._default_font.getbbox(text)[2:]
			width = int(width)
			height = int(height)
		image = Image.new('RGBA', (width + 1, height + 1))
		draw = ImageDraw(image)
		draw.multiline_text((0, 0), text, font=TierList._default_font, align='center')
		return image


class TextBoxTierList(TierList[T]):
	"""Pads out the images from the default implementation of get_item_image, so they are all evenly spaced.

	Hopefully looks a bit less bad."""

	@cached_property
	def images(self) -> Mapping[T, Image.Image]:
		unscaled = super().images
		max_height = max(im.height for im in unscaled.values())
		max_width = max(im.width for im in unscaled.values())
		return {
			item: image_utils.draw_box(
				image_utils.pad_image(
					image, max_width + 2, max_height + 2, (0, 0, 0, 0), centred=True
				)
			)
			for item, image in unscaled.items()
		}


class ImageHaver(Protocol):
	@property
	def image(self) -> Image.Image: ...


class CustomImageTierList(BaseTierList[ImageHaver]):
	"""Tier list for things that have a .image property."""

	def get_item_image(self, item: ImageHaver) -> Image.Image:
		return item.image
