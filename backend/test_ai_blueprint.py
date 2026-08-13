import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend directory is in path
sys.path.append(os.path.dirname(__file__))

# Set dummy environment variables for tests
os.environ["GEMINI_API_KEY"] = "dummy-key-for-testing"

import importlib.util
# Load app.py module explicitly to resolve naming collision with the app/ package folder
backend_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(backend_dir, "app.py")
spec = importlib.util.spec_from_file_location("app_module", app_path)
app_module = importlib.util.module_from_spec(spec)
sys.modules["app_module"] = app_module
spec.loader.exec_module(app_module)
app = app_module.app

from supabase_client import supabase_admin

class TestAIBlueprint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.ctx = app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    @patch('resume_routes.supabase_admin.auth.get_user')
    @patch('app.blueprints.ai.routes.handle_supabase_op')
    @patch('app.blueprints.ai.db_service.handle_supabase_op')
    def test_analyze_resume_endpoint(self, mock_db_service_op, mock_routes_op, mock_get_user):
        # 1. Mock Authentication
        mock_user = MagicMock()
        mock_user.user.id = "mock-user-123"
        mock_get_user.return_value = mock_user

        # 2. Mock Resume retrieval (extracted_text)
        mock_routes_op.return_value = {
            "extracted_text": "Experienced Python Software Engineer with Flask, JavaScript, SQL. Developed web apps.",
            "filename": "test_resume.pdf"
        }

        # 3. Mock Database check for cached analysis & insert (to trigger fallback/mock insert)
        mock_db_service_op.side_effect = lambda callback, fallback: fallback()

        # 4. Trigger analyze-resume POST
        response = self.client.post(
            '/api/ai/analyze-resume',
            headers={"Authorization": "Bearer mock-jwt-token"},
            json={"resume_id": "mock-resume-uuid-456"}
        )

        print("\n=== POST /api/ai/analyze-resume response ===")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.get_json(), indent=2))
        
        self.assertEqual(response.status_code, 201)
        res_json = response.get_json()
        self.assertEqual(res_json["resume_id"], "mock-resume-uuid-456")
        self.assertIn("analysis_results", res_json)
        self.assertEqual(res_json["analysis_results"]["resume_summary"], 
                         "Highly motivated and results-oriented Software Engineer with 2+ years of experience building modern web applications. Proficient in Python, Flask, JavaScript, and React, with a strong focus on clean architecture, performance optimization, and scalable backend design.")

    @patch('resume_routes.supabase_admin.auth.get_user')
    @patch('app.blueprints.ai.db_service.handle_supabase_op')
    def test_history_endpoint(self, mock_db_service_op, mock_get_user):
        # 1. Mock Authentication
        mock_user = MagicMock()
        mock_user.user.id = "mock-user-123"
        mock_get_user.return_value = mock_user

        # 2. Mock DB select for history to return mock list
        mock_db_service_op.side_effect = lambda callback, fallback: fallback()

        # 3. Trigger history GET
        response = self.client.get(
            '/api/ai/history',
            headers={"Authorization": "Bearer mock-jwt-token"}
        )

        print("\n=== GET /api/ai/history response ===")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.get_json(), indent=2))

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.get_json(), list)

if __name__ == '__main__':
    unittest.main()
