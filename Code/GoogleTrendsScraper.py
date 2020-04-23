import pandas as pd
import time
import numpy as np
import math
from selenium import webdriver
from selenium.common import exceptions
import os
import shutil
from datetime import datetime, timedelta
from functools import reduce


def scale_trend(data_daily, data_all, frequency):
    """
    Function that rescales the d
    Args:
        data_daily: pandas.DataFrame of daily trend data
        data_all: pandas.DataFrame of the trend data over the entire sample at a lower frequency
        frequency: frequency of 'data_all'

    Returns: pandas.DataFrame

    """
    # Ensure that all data is numeric:
    data_daily = data_daily.replace('<1', 0.5).astype('float')
    data_all = data_all.replace('<1', 0.5).astype('float')
    # Compute factor by distinguishing between the frequency of the data at the lower frequency (weekly or monthly)
    if frequency == "Weekly":
        factor_dates = pd.date_range(next_weekday(data_daily.index[0], 0),
                                     previous_weekday(data_daily.index[-1], 6), freq='D')
        data_daily_weekly = data_daily.loc[factor_dates].resample('W').sum()
        factor = data_all.loc[data_daily_weekly.index] / data_daily_weekly
    elif frequency == "Monthly":
        factor_dates = pd.date_range(ceil_start_month(data_daily.index[0]),
                                     floor_end_month(data_daily.index[-1]), freq='D')
        data_daily_monthly = data_daily.loc[factor_dates].resample('M', label="left", loffset=timedelta(1)).sum()
        factor = data_all.loc[data_daily_monthly.index] / data_daily_monthly
    # Transform the factor from a pandas.DataFrame to a flat numpy.array
    factor = np.array(factor).flatten()
    # Remove all factor entries for which either of the series is zero
    factor = factor[factor != 0]
    factor = factor[np.isfinite(factor)]
    # Rescale and return the daily trends
    return data_daily*np.median(factor)


def concat_data(data_list, data_all, keywords, frequency):
    """
    Function that merge the DataFrames obtained from different scrapes of GoogleTrends. The DataFrames are collected in
    a list (ordered chronologically), with the last and first observation of two consecutive DataFrames being from the
    same day.
    Args:
        data_list: list of pandas DataFrame objects
        data_all: pandas.DataFrame of trend data over the entire period for the same keywords
        keywords: list of the keywords for which GoogleTrends has been scraped
        frequency:

    Returns: pandas DataFrame with a 'Date' column and a column for each keyword in 'keywords'

    """
    # Remove trend subperiods for which no data has been found
    data_list = [data for data in data_list if data.shape[0] != 0]
    # Rescale the daily trends based on the data at the lower frequency
    data_list = [scale_trend(x, data_all, frequency) for x in data_list]
    # Combine the single trends that were scraped:
    data = reduce((lambda x, y: x.combine_first(y)), data_list)
    # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    # Rename the columns
    data.columns = keywords
    return data


def merge_two_keyword_chunks(data_first, data_second):
    """
    Given two data frame objects with same index and one overlapping column (keyword), a scaling factor
    is determined and the data frames are merged, where the second data frame is rescaled to match the
    scale of the first data set.

    Args:
        data_first: pandas.DataFrame obtained from the csv-file created by GoogleTrends
        data_second: pandas.DataFrame obtained from the csv-file created by GoogleTrends

    Returns: pandas.DataFrame of the merge and re-scaled input pandas.DataFrame

    """
    common_keyword = data_first.columns.intersection(data_second.columns)[0]
    scaling_factor = np.nanmedian(data_first[common_keyword] / data_second[common_keyword])
    data_second = data_second.apply(lambda x: x * scaling_factor)
    data = pd.merge(data_first, data_second.drop(common_keyword, axis=1), left_index=True, right_index=True)
    return data


def merge_keyword_chunks(data_list):
    """
    Merge a list of pandas.DataFrame objects with the same index and one overlapping column by appropriately
    re-scaling.

    Args:
        data_list: list of pandas.DataFrame objects to be merged

    Returns: pandas.DataFrame objects of the merged data sets contained in the input list

    """
    # Iteratively merge the DataFrame objects in the list of data
    data = reduce((lambda x, y: merge_two_keyword_chunks(x, y)), data_list)
    # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    return data


def adjust_date_format(date, format_in, format_out):
    """
    Converts a date-string from one format to another
    Args:
        date: datetime as a string
        format_in: format of 'date'
        format_out: format to which 'date' should be converted

    Returns: date as a string in the new format

    """
    return datetime.strptime(date, format_in).strftime(format_out)


def get_chunks(list_object, chunk_size):
    """
    Generator that divides a list into chunks. If the list is divided in two or more chunks, two consecutive chunks
    have one common element.

    Args:
        list_object: iterable
        chunk_size: size of each chunk as an integer

    Returns: iterable list in chunks with one overlapping element

    """
    size = len(list_object)
    if size <= chunk_size:
        yield list_object
    else:
        chunks_nb = math.ceil(size / chunk_size)
        iter_ints = range(0, chunks_nb)
        for i in iter_ints:
            j = i * chunk_size
            if i + 1 < chunks_nb:
                k = j + chunk_size
                yield list_object[max(j - 1, 0):k]
            else:
                yield list_object[max(j - 1, 0):]


def previous_weekday(date, weekday):
    """
    Function that rounds a date down to the previous date of the desired weekday
    Args:
        date: a datetime.date or datetime.datetime object
        weekday: the desired week day as integer (Monday = 0, ..., Sunday = 6)

    Returns: datetime.date or datetime.datetime object

    """
    delta = date.weekday() - weekday
    if delta < 0:
        delta += 7
    return date + timedelta(days=-int(delta))


def next_weekday(date, weekday):
    """
    Function that rounds a date up to the next date of the desired weekday
    Args:
        date: a datetime.date or datetime.datetime object
        weekday: the desired week day as integer (Monday = 0, ..., Sunday = 6)

    Returns: datetime.date or datetime.datetime object

    """
    delta = weekday - date.weekday()
    if delta < 0:
        delta += 7
    return date + timedelta(days=int(delta))


def ceil_start_month(date):
    """
    Ceil date to the start date of the next month
    Args:
        date: datetime.datetime object

    Returns: datetime.datetime object

    """
    if date.month == 12:
        date = datetime(date.year + 1, 1, 1)
    else:
        date = datetime(date.year, date.month + 1, 1)
    return date


def floor_end_month(date):
    """
    Floor date to the end of the previous month
    Args:
        date: datetime.datetime object

    Returns: datetime.datetime object

    """
    return datetime(date.year, date.month, 1) + timedelta(days=-1)


class GoogleTrendsScraper:
    def __init__(self, sleep=1, path_driver=None, headless=True, date_format='%Y-%m-%d'):
        """
        Constructor of the Google-Scraper-Class
        Args:
            sleep: integer number of seconds where the scraping waits (avoids getting blocked and gives the code time
                    to download the data
            path_driver: path as string to where the chrome driver is located
            headless: boolean indicating whether the browser should be displayed or not
            date_format: format in which the date-strings are passed to the object
            n_overlap: integer number of overlapping observations used to rescale multiple sub-period trends
        """
        # Current directory
        self.dir = os.getcwd()
        # Create a temporary folder in case it does not exist yet
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        # Define the path to the downloaded csv-files (this is where the trends are saved)
        self.filename = 'tmp/multiTimeline.csv'
        # Whether the browser should be opened in headless mode
        self.headless = headless
        # Path to the driver of Google Chrome
        self.path_driver = path_driver
        # Initialize the browser variable
        self.browser = None
        # Sleep time used during the scraping procedure
        self.sleep = sleep
        # Maximal number of consecutive days scraped
        self.max_days = 200
        # Format of the date-strings
        self.date_format = date_format
        # Format of dates used by google
        self._google_date_format = '%Y-%m-%d'
        # Lunch the browser
        self.start_browser()

    def start_browser(self):
        """
        Method that initializes a selenium browser using the chrome driver

        """
        # If the browser is already running, do not start a new one
        if self.browser is not None:
            print('Browser already running')
            pass
        # Options for the browser
        chrome_options = webdriver.ChromeOptions()
        # If the browser should be run in headless mode
        if self.headless:
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('window-size=1920x1080')
        # If no path for the chrome drive is defined, the default is used, i.e. path variables are checked
        if self.path_driver is None:
            self.path_driver = 'chromedriver'
        # Start the browser
        self.browser = webdriver.Chrome(executable_path=self.path_driver, chrome_options=chrome_options)
        # Define the download behaviour of chrome
        # noinspection PyProtectedMember
        self.browser.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        self.browser.execute("send_command", {'cmd': 'Page.setDownloadBehavior', 'params':
            {'behavior': 'allow', 'downloadPath': self.dir + r'\tmp'}})

    def quit_browser(self):
        """
        Method that closes the existing browser

        """
        if self.browser is not None:
            self.browser.quit()
            self.browser = None

    def get_trends(self, keywords, start, end, region=None):
        """
        Function that starts the scraping procedure and returns the Google Trend data.
        Args:
            keywords: list or string of keyword(s)
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string

        Returns: pandas DataFrame with a 'Date' column and a column containing the trend for each keyword in 'keywords'

        """
        # If only a single keyword is given, i.e. as a string and not as a list, put the single string into a list
        if not isinstance(keywords, list):
            keywords = [keywords]
        # Convert the date strings to Google's format:
        start = adjust_date_format(start, self.date_format, self._google_date_format)
        end = adjust_date_format(end, self.date_format, self._google_date_format)
        # Create datetime objects from the date-strings:
        start_datetime = datetime.strptime(start, self._google_date_format)
        end_datetime = datetime.strptime(end, self._google_date_format)
        data_keywords_list = []
        for keywords_i in get_chunks(keywords, 5):
            # Get the trends over the entire sample:
            url_all_i = self.create_url(keywords_i,
                                        previous_weekday(start_datetime, 0), next_weekday(end_datetime, 6),
                                        region)
            data_all_i, frequency_i = self.get_data(url_all_i)
            # If the data for the entire sample is already at the daily frequency we are done. Otherwise we need to
            # get the trends for sub-periods
            if frequency_i == 'Daily':
                data_i = data_all_i
            else:
                # Iterate over the URLs of the sub-periods and retrieve the Google Trend data for each
                data_time_list = []
                for url in self.create_urls_subperiods(keywords_i, start_datetime, end_datetime, region):
                    data_time_list.append(self.get_data(url)[0])
                # Concatenate the so obtained set of DataFrames to a single DataFrame
                data_i = concat_data(data_time_list, data_all_i, keywords_i, frequency_i)
            # Add the data for the current list of keywords to a list
            data_keywords_list.append(data_i)
        # Merge the multiple keyword chunks
        data = merge_keyword_chunks(data_keywords_list)
        # Cut data to return only the desired period:
        data = data.loc[data.index.isin(pd.date_range(start_datetime, end_datetime, freq='D'))]
        return data

    def create_urls_subperiods(self, keywords, start, end, region=None):
        """
        Generator that creates an iterable of URLs that open the Google Trends for a series of sub periods
        Args:
            keywords: list of string keywords
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string

        Returns: iterable of URLs for Google Trends for sub periods of the entire period defined by 'start' and 'end'

        """
        # Calculate number of sub-periods and their respective length:
        num_subperiods = np.ceil(((end - start).days + 1) / self.max_days)
        num_days_in_subperiod = np.ceil(((end - start).days + 1)/num_subperiods)
        for x in range(int(num_subperiods)):
            start_sub = start + timedelta(days=x * num_days_in_subperiod)
            end_sub = start + timedelta(days=(x + 1) * num_days_in_subperiod - 1)
            if end_sub > end:
                end_sub = end
            if start_sub < end:
                yield self.create_url(keywords, start_sub, end_sub, region=region)

    def create_url(self, keywords, start, end, region=None):
        """
        Creates a URL for Google Trends
        Args:
            keywords: list of string keywords
            start: start date as a string
            end: end date as a string
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)

        Returns: string of the URL for Google Trends of the given keywords over the time period from 'start' to 'end'

        """
        # Define main components of the URL
        base = "https://trends.google.com/trends/explore"
        geo = f"geo={region}&" if region is not None else ""
        query = f"q={','.join(keywords)}"
        # Format the datetime objects to strings in the format used by google
        start_string = datetime.strftime(start, self._google_date_format)
        end_string = datetime.strftime(end, self._google_date_format)
        # Define the date-range component for the URL
        date = f"date={start_string}%20{end_string}"
        # Construct the URL
        url = f"{base}?{date}&{geo}{query}"
        return url

    def get_data(self, url):
        """
        Method that retrieves for a specific URL the Google Trend data. Note that this is done by downloading a csv-file
        which is then loaded and stored as a pandas.DataFrame object
        Args:
            url: URL for the trend to be scraped as a string

        Returns: a pandas.DataFrame object containing the trends for the given URL

        """
        # Initialize the button that needs to be pressed to get download the data
        button = None
        # While this button is of type 'None' we reload the browser
        while button is None:
            try:
                # Navigate to the URL
                self.go_to_url(url)
                # Sleep the code by the defined time plus a random number of seconds between 0s and 2s. This should
                # reduce the likelihood that Google detects us as a scraper
                time.sleep(self.sleep + 2 * np.random.rand(1))
                # Try to find the button and click it
                line_chart = self.browser.find_element_by_css_selector("widget[type='fe_line_chart']")
                button = line_chart.find_element_by_css_selector('.widget-actions-item.export')
                button.click()
            except exceptions.NoSuchElementException:
                # If the button cannot be found, try again (load page, ...)
                pass
        # After downloading, wait again to allow the file to be downloaded
        time.sleep(self.sleep)
        # Load the data from the csv-file as pandas.DataFrame object
        data = pd.read_csv(self.filename, skiprows=1)
        # Set date as index:
        if 'Day' in data.columns:
            data.Day = pd.to_datetime(data.Day)
            data = data.set_index("Day")
            frequency = 'Daily'
        elif 'Week' in data.columns:
            data.Week = pd.to_datetime(data.Week)
            data = data.set_index("Week")
            frequency = 'Weekly'
        else:
            data.Month = pd.to_datetime(data.Month)
            data = data.set_index("Month")
            frequency = 'Monthly'
        # Sleep again
        time.sleep(self.sleep)
        # Delete the file
        while os.path.exists(self.filename.replace('/', '\\')):
            try:
                os.remove(self.filename.replace('/', '\\'))
            except:
                pass
        return data, frequency

    def go_to_url(self, url):
        """
        Method that navigates in the browser to the given URL
        Args:
            url: URL to which we want to navigate as a string

        """
        if self.browser is not None:
            self.browser.get(url)
        else:
            print('Browser is not running')

    def __del__(self):
        """
        When deleting an instance of this class, delete the temporary file folder and close the browser

        """
        shutil.rmtree('tmp')
        self.quit_browser()
