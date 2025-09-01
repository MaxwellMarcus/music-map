"""
Earth Tone Color Palette for Musical Artist Visualization
A collection of natural, earthy colors for cluster visualization
"""

# Earth tone color palette - natural, warm colors
EARTH_TONE_PALETTE = [
    '#8B4513',  # Saddle Brown
    '#A0522D',  # Sienna
    '#CD853F',  # Peru
    '#D2691E',  # Chocolate
    '#B8860B',  # Dark Goldenrod
    '#DAA520',  # Goldenrod
    '#F4A460',  # Sandy Brown
    '#DEB887',  # Burly Wood
    '#D2B48C',  # Tan
    '#BC8F8F',  # Rosy Brown
    '#8FBC8F',  # Dark Sea Green
    '#556B2F',  # Dark Olive Green
    '#6B8E23',  # Olive Drab
    '#808000',  # Olive
    '#BDB76B',  # Dark Khaki
    '#F0E68C',  # Khaki
    '#9ACD32',  # Yellow Green
    '#6B4423',  # Dark Brown
    '#8B7355',  # Dark Tan
    '#A0522D',  # Sienna
    '#CD5C5C',  # Indian Red
    '#B22222',  # Fire Brick
    '#DC143C',  # Crimson
    '#8B0000',  # Dark Red
    '#2F4F4F',  # Dark Slate Gray
    '#696969',  # Dim Gray
    '#708090',  # Slate Gray
    '#778899',  # Light Slate Gray
    '#4682B4',  # Steel Blue
    '#5F9EA0',  # Cadet Blue
    '#20B2AA',  # Light Sea Green
    '#48D1CC',  # Medium Turquoise
    '#40E0D0',  # Turquoise
    '#7FFFD4',  # Aquamarine
    '#66CDAA',  # Medium Aquamarine
    '#3CB371',  # Medium Sea Green
    '#2E8B57',  # Sea Green
    '#228B22',  # Forest Green
    '#32CD32',  # Lime Green
    '#90EE90',  # Light Green
    '#98FB98',  # Pale Green
    '#8FBC8F',  # Dark Sea Green
    '#556B2F',  # Dark Olive Green
    '#6B8E23',  # Olive Drab
    '#BDB76B',  # Dark Khaki
    '#F0E68C',  # Khaki
    '#EEE8AA',  # Pale Goldenrod
    '#F5DEB3',  # Wheat
    '#DEB887',  # Burly Wood
    '#D2B48C',  # Tan
    '#BC8F8F',  # Rosy Brown
    '#F4A460',  # Sandy Brown
    '#DAA520',  # Goldenrod
    '#B8860B',  # Dark Goldenrod
    '#CD853F',  # Peru
    '#D2691E',  # Chocolate
    '#A0522D',  # Sienna
    '#8B4513',  # Saddle Brown
    '#654321',  # Dark Brown
    '#8B7355',  # Dark Tan
    '#A0522D',  # Sienna
    '#CD5C5C',  # Indian Red
    '#B22222',  # Fire Brick
    '#DC143C',  # Crimson
    '#8B0000',  # Dark Red
    '#2F4F4F',  # Dark Slate Gray
    '#696969',  # Dim Gray
    '#708090',  # Slate Gray
    '#778899',  # Light Slate Gray
    '#4682B4',  # Steel Blue
    '#5F9EA0',  # Cadet Blue
    '#20B2AA',  # Light Sea Green
    '#48D1CC',  # Medium Turquoise
    '#40E0D0',  # Turquoise
    '#7FFFD4',  # Aquamarine
    '#66CDAA',  # Medium Aquamarine
    '#3CB371',  # Medium Sea Green
    '#2E8B57',  # Sea Green
    '#228B22',  # Forest Green
    '#32CD32',  # Lime Green
    '#90EE90',  # Light Green
    '#98FB98',  # Pale Green
    '#8FBC8F',  # Dark Sea Green
    '#556B2F',  # Dark Olive Green
    '#6B8E23',  # Olive Drab
    '#BDB76B',  # Dark Khaki
    '#F0E68C',  # Khaki
    '#EEE8AA',  # Pale Goldenrod
    '#F5DEB3',  # Wheat
    '#DEB887',  # Burly Wood
    '#D2B48C',  # Tan
    '#BC8F8F',  # Rosy Brown
    '#F4A460',  # Sandy Brown
    '#DAA520',  # Goldenrod
    '#B8860B',  # Dark Goldenrod
    '#CD853F',  # Peru
    '#D2691E',  # Chocolate
    '#A0522D',  # Sienna
    '#8B4513',  # Saddle Brown
]

def get_earth_tone_color(index):
    """Get an earth tone color by index, cycling through the palette"""
    return EARTH_TONE_PALETTE[index % len(EARTH_TONE_PALETTE)]

def get_earth_tone_colors_for_clusters(cluster_ids):
    """Generate earth tone colors for a list of cluster IDs"""
    color_map = {}
    for i, cluster_id in enumerate(cluster_ids):
        color_map[cluster_id] = get_earth_tone_color(i)
    return color_map
