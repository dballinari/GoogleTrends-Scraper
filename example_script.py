from src.GoogleTrendsScraper import GoogleTrendsScraper

gts = GoogleTrendsScraper(
    sleep=2, path_driver='C:/Users/danie/chromedriver/chromedriver.exe', headless=False)
data = gts.get_trends('foo', '2018-07-02', '2019-04-02', 'US')

print(data)

del gts
