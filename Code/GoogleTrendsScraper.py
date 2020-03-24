import pandas as pd
import time
import numpy as np
import math
from selenium import webdriver
from selenium.common import exceptions
import os
from datetime import datetime, timedelta
from functools import reduce


def merge_two_trends(data_first, data_second):
    """
    Function that merges two panda data frames obtained from GoogleTrends. Last observation of 'data_first' and first
    observation of 'data_second' have to be overlapping (same date).
    Args:
        data_first: pandas.DataFrame obtained from the csv-file created by GoogleTrends
        data_second: pandas.DataFrame obtained from the csv-file created by GoogleTrends

    Returns: single pandas.DataFrame where 'data_second' is appended to 'data_first', and the two data sets are scaled
    to be on the same scale.

    """

    # In some cases, GoogleTrends reports very small values as '<1'. These values are replaced by 0.5 and the data-type
    # changed to 'float'. Note that the first column contains the dates as strings and therefore is not modified.
    data_first = data_first.replace('<1', 0.5).astype('float')
    data_second = data_second.replace('<1', 0.5).astype('float')
    # Find overlapping observations:
    common_obs = data_first.index.intersection(data_second.index)
    # Calculate the scaling factor, by computing a ratio of the overlapping observations
    factor = np.nanmedian(data_first.loc[common_obs, :]/data_second.loc[common_obs, :])
    # Rescale the second data set with the factor, such that all observations are on a comparable scale
    data_second = data_second * factor
    # Combine the data sets and for overlapping observations, keep only the first one
    return data_first.combine_first(data_second)


def merge_data(data_list, keywords):
    """
    Function that merge the DataFrames obtained from different scrapes of GoogleTrends. The DataFrames are collected in
    a list (ordered chronologically), with the last and first observation of two consecutive DataFrames being from the
    same day.
    Args:
        data_list: list of pandas DataFrame objects
        keywords: list of the keywords for which GoogleTrends has been scraped

    Returns: pandas DataFrame with a 'Date' column and a column for each keyword in 'keywords'

    """
    # Iteratively merge the DataFrame objects in the list of data
    data = reduce((lambda x, y: merge_two_trends(x, y)), data_list)
    # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    # Rename the columns
    data.columns = keywords
    return data


def merge_two_keyword_chunks(data_first, data_second):
    common_keyword = data_first.columns.intersection(data_second.columns)[0]
    scaling_factor = np.nanmedian(data_first[common_keyword] / data_second[common_keyword])
    data_second = data_second.apply(lambda x: x * scaling_factor)
    data = pd.merge(data_first, data_second.drop(common_keyword, axis=1), left_index=True, right_index=True)
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
    Generator that divides a list into chunks
    Args:
        list_object: iterable
        chunk_size: size of each chunk as an integer

    Returns: iterable in chunks

    """
    size = len(list_object)
    if size <= chunk_size:
        yield list_object
    chunks_nb = math.ceil(size / chunk_size)
    iter_ints = range(0, chunks_nb)
    for i in iter_ints:
        j = i * chunk_size
        if i + 1 < chunks_nb:
            k = j + chunk_size
            yield list_object[max(j - 1, 0):k]
        else:
            yield list_object[max(j - 1, 0):]


class GoogleTrendsScraper:
    def __init__(self, sleep=1, path_driver=None, headless=True, date_format='%Y-%m-%d', n_overlap=5):
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
        self.max_days = 240
        # Format of the date-strings
        self.date_format = date_format
        # Format of dates used by google
        self._google_date_format = '%Y-%m-%d'
        # Number of overlapping observations:
        self.n_overlap = n_overlap
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
            # Iterate over the URLs of the sub-periods and retrieve the Google Trend data for each
            data_time_list = []
            for url in self.create_urls(keywords_i, start_datetime, end_datetime, region):
                data_time_list.append(self.get_data(url))
            # Merge the so obtained set of DataFrames to a single DataFrame
            data_keywords_list.append(merge_data(data_time_list, keywords_i))

        data = reduce((lambda x, y: merge_two_keyword_chunks(x, y)), data_keywords_list)
        return data

    def create_urls(self, keywords, start, end, region=None):
        """
        Generator that creates an iterable of URLs that open the Google Trends for a series of sub periods
        Args:
            keywords: list of string keywords
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string

        Returns: iterable of URLs for Google Trends for sub periods of the entire period defined by 'start' and 'end'

        """
        # Define main components of the URL
        base = "https://trends.google.com/trends/explore"
        geo = f"geo={region}&" if region is not None else ""
        query = f"q={','.join(keywords)}"
        # Calculate number of sub-periods and their respective length
        num_subperiods = np.ceil((end - start).days / self.max_days)
        length_subperiods = np.ceil((end - start).days / num_subperiods)

        for x in range(int(num_subperiods)):
            start_sub = start + timedelta(days=x * length_subperiods)
            end_sub = start + timedelta(days=(x + 1) * length_subperiods + 4)
            # It might be that the end of the sub-period exceeds the actual end of the trend period. In that case, we
            # let this period end at the date defined by 'end'
            if end_sub > end:
                end_sub = end
            # Format the datetime objects to strings in the format used by google
            start_sub_string = datetime.strftime(start_sub, self._google_date_format)
            end_sub_string = datetime.strftime(end_sub, self._google_date_format)
            # Define the date-range component for the URL
            date = f"date={start_sub_string}%20{end_sub_string}"
            # Construct the URL
            url = f"{base}?{date}&{geo}{query}"
            yield url

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
        data = data.set_index("Day")
        # Sleep again
        time.sleep(self.sleep)
        # Delete the file
        os.remove(self.filename.replace('/', '\\'))
        return data

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
        os.rmdir('tmp')
        self.quit_browser()
