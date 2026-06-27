import configparser
from pathlib import Path

appconfig = configparser.ConfigParser()
appconfig.read(str(Path(__file__).parent)+'/config.ini')