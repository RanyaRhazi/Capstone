"""API tests. Run with: python -m unittest -v  (after python train.py)."""
import unittest

from sklearn.datasets import load_digits

from flaskapp import app


class PredictEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.digits = load_digits()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_predicts_known_digit(self):
        image = self.digits.data[0].tolist()  # a "0"
        response = self.client.post("/predict", json={"image": image})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], [int(self.digits.target[0])])

    def test_accepts_8x8_nested_list(self):
        image = self.digits.images[1].tolist()  # a "1", as 8 rows of 8
        response = self.client.post("/predict", json={"image": image})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], [int(self.digits.target[1])])

    def test_rejects_wrong_length(self):
        response = self.client.post("/predict", json={"image": [0.0] * 10})
        self.assertEqual(response.status_code, 400)

    def test_rejects_missing_field(self):
        response = self.client.post("/predict", json={"pixels": [0.0] * 64})
        self.assertEqual(response.status_code, 400)

    def test_rejects_non_numeric(self):
        response = self.client.post("/predict", json={"image": ["a"] * 64})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
