import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent import gtm_agent


class ProspectDataSanitizationTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_and_disqualification_fields(self):
        record = {
            "name": "Alex Example",
            "email": "alex@example.com",
            "disqualified": True,
            "billing_qualification": {"tax_id": "123-45-6789"},
            "annual_revenue": 1000000,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-TEST"})

        self.assertTrue(result["found"])
        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-TEST",
                "name": "Alex Example",
                "email": "alex@example.com",
                "disqualified": True,
            },
        )
        self.assertTrue(result["prospect"]["disqualified"])

    def test_new_profile_excludes_billing_qualification(self):
        prospect_id = "LEAD-NEW-PROFILE-TEST"
        record = {
            "name": "Alex Example",
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "123-45-6789"},
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value={}),
            patch.object(data_service, "fetch_tech_stack", return_value=[]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})

        self.assertTrue(result["found"])
        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("billing_qualification", save_profile.call_args.args[1])

    def test_cached_profile_is_sanitized_and_replaced(self):
        prospect_id = "LEAD-CACHED-PROFILE-TEST"
        data_service._PROFILES[prospect_id] = {
            "prospect_id": prospect_id,
            "name": "Alex Example",
            "billing_qualification": {"tax_id": "123-45-6789"},
        }
        self.addCleanup(data_service._PROFILES.pop, prospect_id, None)

        first_result = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        second_result = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})

        self.assertNotIn("billing_qualification", first_result["prospect_profile"])
        self.assertNotIn("billing_qualification", second_result["prospect_profile"])
        self.assertNotIn("billing_qualification", data_service._PROFILES[prospect_id])

    def test_scoring_model_receives_only_scoring_fields(self):
        prospect_id = "LEAD-SCORING-TEST"
        model = Mock()
        model.invoke.return_value = SimpleNamespace(model_dump=lambda: {"score": 80})
        profile = {
            "prospect_id": prospect_id,
            "name": "Alex Example",
            "annual_revenue": 1000000,
            "tech_stack": ["old stack"],
            "account_details": {"industry": "software"},
            "engagement_history": ["attended webinar"],
            "billing_qualification": {"tax_id": "123-45-6789"},
            "disqualified": True,
            "arbitrary_extra": "not for scoring",
        }
        offering = {
            "required_tech_stack": ["python"],
            "min_annual_revenue": 100,
            "description": "A test offering",
        }
        with (
            patch.object(gtm_agent, "_scoring_llm", model),
            patch.object(data_service, "fetch_tech_stack", return_value=["python"]),
        ):
            gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        serialized_user = model.invoke.call_args.args[0][1]["content"]
        serialized_profile = json.loads(serialized_user.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            set(serialized_profile),
            {
                "prospect_id", "name", "annual_revenue", "tech_stack",
                "account_details", "engagement_history",
            },
        )
        self.assertEqual(serialized_profile["tech_stack"], ["python"])
        self.assertNotIn("billing_qualification", serialized_user)


if __name__ == "__main__":
    unittest.main()
