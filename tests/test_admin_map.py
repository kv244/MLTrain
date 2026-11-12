"""Admin login and public country map. Skipped when app.py's heavy imports (torch/openvino) are missing."""
import os
import sys
import unittest

os.environ["ADMIN_TOKEN"] = "unit-test-token"
os.environ["SESSION_COOKIE_SECURE"] = "0"
os.environ["RATE_LIMIT"] = "100000 per minute"
os.environ.pop("GCP_PROJECT_ID", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import app as appmod
except Exception as exc:  # pragma: no cover
    appmod = None
    _why = str(exc)


@unittest.skipIf(appmod is None, "app.py could not be imported here")
class AdminAndMap(unittest.TestCase):
    def setUp(self):
        self.c = appmod.app.test_client()

    def test_login_flow(self):
        self.assertIn(b"Sign in", self.c.get("/admin").data)
        self.assertEqual(self.c.post("/admin", data={"token": "nope"}).status_code, 401)
        r = self.c.post("/admin", data={"token": "unit-test-token"})
        self.assertEqual(r.status_code, 302)
        self.assertIn(b"Analytics", self.c.get("/admin").data)
        self.c.post("/admin/logout")
        self.assertIn(b"Sign in", self.c.get("/admin").data)

    def test_legacy_token_url_redirects_without_token(self):
        r = self.c.get("/admin/unit-test-token")
        self.assertEqual(r.status_code, 302)
        self.assertNotIn("unit-test-token", r.headers["Location"])
        self.assertEqual(self.c.get("/admin/wrong").status_code, 404)

    def test_map_is_public_and_aggregated(self):
        self.assertEqual(self.c.get("/map").status_code, 200)

        class Q:
            def result(self):
                return [{"country": "Romania", "visitors": 3, "games": None}]

        class Cl:
            def query(self, sql):
                assert "ip_address" not in sql
                return Q()

        appmod.bigquery_tracker._enabled = True
        appmod.bigquery_tracker._client = Cl()
        appmod.bigquery_tracker._table_ref = "p.d.t"
        appmod._country_cache.update(ts=0.0, data=None)
        try:
            d = self.c.get("/api/country_stats").get_json()
        finally:
            appmod.bigquery_tracker._enabled = False
            appmod._country_cache.update(ts=0.0, data=None)
        self.assertEqual(d["countries"], [{"country": "Romania", "visitors": 3, "games": 0}])


if __name__ == "__main__":
    unittest.main()
