# Google Trend scraper
## Introduction
Scraper for [GoogleTrends](https://trends.google.com/trends/?geo=US) based on `selenium`. 
For a given keyword, time range and region, the scraping tool collects the GoogleTrend data. This
data is scaled ranging from 0 to 100, where the day with the highest number of searches is set to 
100\. Note that Google estimates the volume of searches based on a random sample of search-queries.
  
## Set-up
The scraper is constructed using Python 3.7 and is based on `selenium` and `pandas`. The required 
modules can be installed by navigating to the root of this project and running 
`pip install -r requirements.txt`.

Moreover, Selenium needs a browser-driver, e.g. [Google Chrome dirver](https://sites.google.com/a/chromium.org/chromedriver/downloads).
The path to the driver's location has to be added to the system path variables or 
indicated when initializing the `GoogleTrendsScraper` object (see below) 

## Usage
First, a `GoogleTrendsScraper` object needs to be initialized with (optional) parameters, such as
the `sleep` time used to when long time windows are scraped, the `headless` option which defines
whether the browser opened by Selenium is visible. In a second step, the trends for a specific 
keyword, time range and region can be obtained by running the `GoogleTrendsScrpaer.get_trends`
method. 

```python
from Code.GoogleTrendsScraper import GoogleTrendsScraper

gts = GoogleTrendsScraper(sleep=2, path_driver='path/to/driver.exe', headless=True)
data = gts.get_trends('foo', '2018-01-01', '2019-03-31', 'US')

del gts
```


## Implementation details
The scraper opens the GoogleTrends web-page and runs the search. In order to obtain the data, the 
download-button is pressed and a csv-file containing the trends is downloaded. The data is then 
loaded into Python as a `pandas.DataFrame` object. 

GoogleTrends limits the range of consecutive daily observations displayed. For this reason, the 
scraper divides the total time range is sub-periods and downloads the data separately for each of
them. In order to combine the data to a single time series, we download the data for the entire sample
which will be at a lower frequency (weekly or monthly) and use this information to re-scale the 
daily trends collected in sub-periods. More precisely, for each sub-group, we aggregate the data to the
lower frequency and build a ratio between the data for the entire sample and the sub-period data. The rescaling
factor is then defined as the median of these ratios over the given sub-period.

Note that, theoretically, over a single sub-period, the ratio should be constant. However, do to the fact that 
the trends are always based on a random sample, this holds not true. That is why, we define the factor as the 
median of the ratios.