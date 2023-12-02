from src.GoogleTrendsScraper import GoogleTrendsScraper
import os

gts = GoogleTrendsScraper(sleep=2, path_driver=os.environ['CHROMEDRIVER'], headless=True)
data = gts.get_trends('foo', '2018-07-02', '2019-04-02', 'US')

print(data)

del gts
