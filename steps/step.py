# -*- coding: utf-8 -*-
import datetime
from abc import abstractmethod
from warnings import simplefilter

from dotenv import load_dotenv

from utils.config import Config

simplefilter("ignore", category=RuntimeWarning)


class Step:
    """Abstract class outlining what it means to be a Step
    A valid Step has a `run` method to run the code it contains, as well as a valid name and generic methods to save
    and load results.
    """

    def __init__(self, config_path: str) -> None:

        load_dotenv()

        self.date_to_save = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # initialize config
        self.config_init(config_path)

    def config_init(self, config_path):
        self.config = Config(config_path).read()

    @property
    def name(self) -> str:
        """The name of the step instance"""
        return self._name

    @abstractmethod
    def run(self, *args, **kwargs):
        """Runs the step"""
        raise NotImplementedError()

    @abstractmethod
    def save_output(self, *args, **kwargs) -> None:
        """save the output(s) of the step"""
        pass

    @classmethod
    @abstractmethod
    def load_output(cls, *args, **kwargs):
        """Loads the output(s) of the step"""
        pass
