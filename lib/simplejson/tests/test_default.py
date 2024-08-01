<<<<<<< HEAD
from unittest import TestCase

import simplejson as json

class TestDefault(TestCase):
    def test_default(self):
        self.assertEqual(
            json.dumps(type, default=repr),
            json.dumps(repr(type)))
=======
from unittest import TestCase

import simplejson as json

class TestDefault(TestCase):
    def test_default(self):
        self.assertEqual(
            json.dumps(type, default=repr),
            json.dumps(repr(type)))
>>>>>>> 626e7afc02230297b6f553675ea1c32c29971314
