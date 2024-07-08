Library that makes tier list images, given some objects and scores for each of them. Originally it was made for Super Smash Bros. Ultimate characters but should be able to do anything that can be scored/ranked.

# TODO:
- Actual readme
- Implement basic command line interface (and then don't call it a library in this readme)
- Make sklearn (should only be needed for KMeans) and matplotlib (should only be needed for colour maps) optional dependencies
	- The default for tiers should be some magic value that tries to use kmeans_tierer but falls back to quantile_tierer if sklearn is not available
- Categorical tier list values (or basically just providing predefined tiers, and not scores and not computing which tiers to go in)
- Provide strings (as tier names) as tiers argument to BaseTierList
- Option to draw the item's individual ranking/position/whatsitmacallit on its image
- tier_names could also be a function that generates them, since with num_tiers = 'auto' we don't know how many we want