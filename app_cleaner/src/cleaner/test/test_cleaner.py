import unittest
import pandas as pd
from cleaner.core import format

class TestFormat(unittest.TestCase):
    def test_format_downloads(self):
        data = [[
            "Coloring book moana", 
            "ART_AND_DESIGN",
            3.9,
            "967",
            "14M",
            "500,000+",
            "Free",
            "0",
            "Everyone",
            "Art & Design;Pretend Play",
            "January 15, 2018",
            "2.0.0",
            "4.0.3 and up"
        ]]

        input = pd.DataFrame(data, columns=[
            "App",
            "Category",
            "Rating",
            "Reviews",
            "Size",
            "Installs",
            "Type",
            "Price",
            "Content Rating",
            "Genres",
            "Last Updated",
            "Current Ver",
            "Android Ver"
        ])

        output = format.normalizeFields(input)
        self.assertEqual(output["Installs"].values[0], 500000)



if __name__ == '__main__':
    unittest.main()
