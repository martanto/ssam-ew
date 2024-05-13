from . import magma
from .ssam import SsamEW

from pkg_resources import get_distribution

__version__ = get_distribution("ssam-ew").version
__author__ = "Martanto"
__author_email__ = "martanto@live.com"
__license__ = "MIT"

__copyright__ = "Copyright (c) 2024"
__url__ = "https://github.com/martanto/ssam-ew"

__all__ = [
    'SsamEW',
    'magma',
]