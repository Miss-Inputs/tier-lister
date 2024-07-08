import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy
import pandas
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from tier_lister.tiers import Tiers


@dataclass(frozen=True)
class KMeansAutoTiers(Tiers):
	"""Has additional info about the KMeans process"""
	centroids: Mapping[int, float]
	inertia: float
	kmeans_iterations: int


def _cluster_loss(tiers: Tiers, desired_size: float | None) -> float:
	"""Loss function that gives a number closer to 0 for more balanced clusters (containing similar number of elements), kinda"""
	sizes = tiers.tier_ids.value_counts()
	if not desired_size:
		desired_size = sizes.mean()
	diffs = sizes - desired_size
	diffs[diffs < 0] *= (
		2  # This should penalize harder results that end up with small clusters such as 1 item (I think) so you don't end up with 9999 tiers (I think)
	)
	return float(
		(diffs**2).sum()
	)  # Technically return value is numpy.float64 or whatever (but type hinted as Any) and the mypy warning was bugging me

def find_best_clusters(scores: 'pandas.Series[float]') -> KMeansAutoTiers:
	"""Tries to find a value for n_clusters that gives cluster sizes close to sqrt(number of tier items)
	:raises RuntimeError: if it somehow doesn't find anything"""
	best_loss = numpy.inf
	best = None
	# KMeans is invalid with <2 clusters, and wouldn't really make sense with more than the number of unique values
	sqrt = numpy.sqrt(scores.size)
	for i in range(2, scores.nunique()):
		tiers = kmeans_tierer(scores, i)
		loss = _cluster_loss(tiers, sqrt)
		if loss > best_loss:
			continue
		if tiers.kmeans_iterations == 300:
			# KMeans didn't like this and cooked too hard, give up
			continue
		if tiers.tier_ids.nunique() < len(tiers.centroids):
			# KMeans gave us weird results, give up
			continue
		best = tiers
		best_loss = loss
	if not best:
		raise RuntimeError('oh no')
	return best


def kmeans_tierer(
	scores: 'pandas.Series[float]', num_tiers: int | Literal['auto']
) -> KMeansAutoTiers:
	"""Separate scores into tiers with k-means clustering. Ensures tier numbers are monotonically increasing."""
	if num_tiers == 'auto':
		return find_best_clusters(scores)
	kmeans = KMeans(num_tiers, n_init='auto', random_state=0)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', ConvergenceWarning)
		labels = kmeans.fit_predict(scores.to_numpy().reshape(-1, 1))
	raw_tiers = pandas.Series(labels, index=scores.index, name='tiers')
	# The numbers in raw_tiers are just random values for the sake of being distinct, we are already sorted by score, so have ascending tiers instead
	mapping = {c: i for i, c in enumerate(raw_tiers.unique())}
	tiers = raw_tiers.map(mapping)
	centroids = (
		pandas.Series(kmeans.cluster_centers_.squeeze(), name='centroids')
		.rename(mapping)
		.sort_index()
		.to_dict()
	)
	return KMeansAutoTiers(tiers, centroids, kmeans.inertia_, kmeans.n_iter_)
