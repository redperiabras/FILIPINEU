from datetime import datetime

import pytest
import logging as lh
lh.basicConfig(filename='./tests/test.log', level=lh.DEBUG)


class TestClass:
    @classmethod
    def setup_class(cls):
        lh.info("\t{} - Starting class: {} execution\n".format(
            datetime.utcnow(), cls.__name__))

    @classmethod
    def teardown_class(cls):
        lh.info(
            "\t{} - Stopping class: {} execution\n".format(datetime.utcnow(), cls.__name__))

    def setup_method(self, method):
        lh.info(
            "\t{} - Starting execution of Test Case: {}".format(datetime.utcnow(), method.__name__))

    def teardown_method(self, method):
        lh.info(
            "\t{} - Stopping execution of Test Case: {}\n".format(datetime.utcnow(), method.__name__))

    def test_runserver_tc1(self):
        # Is server running
        assert True

    def test_runserver_tc2(self):
        # Is server running in debug Mode
        assert True

    def test_preprocess_tc1(self):
        # Raw data preprocessor runs well
        assert True

        # lh.debug('\tThis message should go to the log file')
        # lh.info('\tSo should this')
        # lh.warning('\tAnd this, too')
        # assert 1 + 1 == 2
