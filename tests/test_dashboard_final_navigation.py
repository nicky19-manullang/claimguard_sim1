import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard_final.py"


class DashboardFinalNavigationTests(unittest.TestCase):
    def test_landing_page_offers_one_entry_into_claim_review(self):
        app = AppTest.from_file(DASHBOARD, default_timeout=90).run()

        self.assertFalse(app.exception)
        self.assertEqual([button.label for button in app.button], ["Mulai pemeriksaan klaim"])
        landing_copy = "\n".join(str(element.value) for element in app.markdown)
        self.assertNotIn("Pilih jalur analisis", landing_copy)
        self.assertNotIn("Model ML", landing_copy)
        self.assertNotIn("Analisis V2", landing_copy)

        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["nav_menu"], "Pemeriksaan klaim")
        self.assertTrue(any("Antrean pemeriksaan klaim" in str(item.value) for item in app.subheader))
        self.assertTrue(any(item.label == "Cara mengelompokkan klaim" for item in app.selectbox))
        self.assertFalse(any(item.label == "Pilih jalur analisis" for item in app.selectbox))


if __name__ == "__main__":
    unittest.main()
