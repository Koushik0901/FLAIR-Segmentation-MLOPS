import unittest
import yaml
import os
import json

class TestConf(unittest.TestCase):
    def test_config(self):
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config

