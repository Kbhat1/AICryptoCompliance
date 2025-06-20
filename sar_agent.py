# Crypto Compliance LLM Agent
# This script provides a small example of how one might integrate the OpenAI API
# with Chainalysis Data Solutions to investigate a KYT alert and gather
# information for a SAR narrative. It uses OpenAI's function calling feature to
# expose Chainalysis queries to the language model.

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import openai
from chainalysis import DataSolutionsClient


class ChainalysisWrapper:
    """Wrapper around the Chainalysis Data Solutions SDK."""

    def __init__(self, api_key: str):
        self.client = DataSolutionsClient(api_key=api_key)

    def get_kyt_alert(self, alert_id: str) -> Dict[str, Any]:
        """Fetch KYT alert details.

        This example uses a placeholder query. Replace the SQL with the
        appropriate table and fields from Chainalysis Data Solutions.
        """
        query = (
            "SELECT * FROM kyt.alerts WHERE alert_id = :alert_id"
        )
        response = self.client.sql.transactional_query(
            query, parameters={"alert_id": alert_id}
        )
        return response.json()

    def trace_address(self, address: str) -> Dict[str, Any]:
        """Example address tracing query.

        Replace the SQL with the relevant Chainalysis query to perform address
        tracing or exposure lookup.
        """
        query = (
            "SELECT * FROM traces WHERE address = :address LIMIT 100"
        )
        response = self.client.sql.analytical_query(
            query, parameters={"address": address}
        )
        return response.json()


class CryptoComplianceAgent:
    """Agent that drives an investigation using OpenAI function calling."""

    def __init__(self, openai_key: str, chainalysis_key: str):
        openai.api_key = openai_key
        self.chainalysis = ChainalysisWrapper(chainalysis_key)

    def _functions(self) -> List[Dict[str, Any]]:
        """Return OpenAI function specs."""
        return [
            {
                "name": "get_kyt_alert",
                "description": "Fetch a KYT alert's details from Chainalysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_id": {"type": "string", "description": "Alert identifier"}
                    },
                    "required": ["alert_id"],
                },
            },
            {
                "name": "trace_address",
                "description": "Trace an on-chain address using Chainalysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "string", "description": "Cryptocurrency address"}
                    },
                    "required": ["address"],
                },
            },
        ]

    def run(self, alert_id: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a crypto compliance analyst. Gather information "
                    "required for a SAR narrative using the available tools."
                ),
            },
            {"role": "user", "content": f"Investigate KYT alert {alert_id}."},
        ]

        while True:
            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=messages,
                functions=self._functions(),
                function_call="auto",
            )

            choice = response["choices"][0]
            if choice.get("finish_reason") == "stop":
                return choice["message"]["content"]

            if choice.get("finish_reason") == "function_call":
                func_name = choice["message"]["function_call"]["name"]
                arguments = json.loads(choice["message"]["function_call"]["arguments"])
                if func_name == "get_kyt_alert":
                    result = self.chainalysis.get_kyt_alert(arguments["alert_id"])
                elif func_name == "trace_address":
                    result = self.chainalysis.trace_address(arguments["address"])
                else:
                    result = {"error": f"Unknown function {func_name}"}

                messages.append(choice["message"])
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(result),
                })
                continue


def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    chainalysis_key = os.environ.get("CHAINALYSIS_API_KEY")
    alert_id = os.environ.get("KYT_ALERT_ID")

    if not (openai_key and chainalysis_key and alert_id):
        raise SystemExit(
            "Please set OPENAI_API_KEY, CHAINALYSIS_API_KEY, and KYT_ALERT_ID"
        )

    agent = CryptoComplianceAgent(openai_key, chainalysis_key)
    narrative = agent.run(alert_id)
    print(narrative)


if __name__ == "__main__":
    main()
