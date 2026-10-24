import boto3

session = boto3.Session(
    profile_name="hackathon",
    region_name="us-east-1"
)

client = session.client("bedrock-runtime")

response = client.converse(
    modelId="deepseek.v3.2",
    messages=[
        {
            "role": "user",
            "content": [{
                "text": (
                    "Analyze this fictional wildfire report: "
                    "Fire crews report visible flames in West Ridge. "
                    "Return the location and severity."
                )
            }]
        }
    ],
    inferenceConfig={
        "maxTokens": 300
    }
)

print(response["output"]["message"]["content"])
