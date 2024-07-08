import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Protocol

import pandas


def scale_to_range(s: pandas.Series, min_out: float = 0, max_out: float = 1.0):
	min_in = s.min(skipna=True)
	max_in = s.max(skipna=True)
	range_in = max_in - min_in
	if not range_in:
		range_in = 1.0

	return ((max_out - min_out) * ((s - min_in) / range_in)) + min_out

@dataclass(frozen=True)
class Tiers:
	"""Holds a mapping of item -> tier ID and associated things. Can be automatically generated."""

	tier_ids: 'pandas.Series[int]'
	"""Index = the same one that was passed in for scores, values = which tier ID is assigned to which thing"""
	centroids: Mapping[int, float] | None
	"""Tier ID -> centre, or None if not applicable."""

	@cached_property
	def scaled_centroids(self) -> Mapping[int, float] | None:
		"""Scale centroids between 0.0 and 1.0, used for colour mapping, or None if this is not applicable"""
		if not self.centroids:
			return None
		values = scale_to_range(pandas.Series(list(self.centroids.values())))
		# self.centroids does not necessarily have linear ascending keys
		return dict(zip(self.centroids.keys(), values, strict=True))


class AutoTierer(Protocol):
	"""Something that takes tiers + values and returns a Tiers."""

	def __call__(
		self, scores: 'pandas.Series[float]', num_tiers: int | Literal['auto']
	) -> Tiers: ...


def quantile_tierer(scores: 'pandas.Series[float]', num_tiers: int | Literal['auto']):
	"""Tiers scores into equal quantiles."""
	if num_tiers == 'auto':
		num_tiers = math.isqrt(scores.size)
	tiers = pandas.cut(scores, num_tiers, labels=list(reversed(range(num_tiers)))).astype(int)
	centroids = scores.groupby(tiers).mean()
	return Tiers(tiers, centroids.to_dict())
