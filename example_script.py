from Code.GoogleTrendsScraper import GoogleTrendsScraper

gts = GoogleTrendsScraper(sleep=2, path_driver='C:/Users/danie/chromedriver/chromedriver.exe', headless=False)
data = gts.get_trends('aapl', '2018-01-01', '2019-03-31', 'US')

del gts
