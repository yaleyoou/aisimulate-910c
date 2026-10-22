"""Make the isolated native build visible to every pytest worker."""

import aisimulate

aisimulate.__path__.append("/tmp/aisim-native")
