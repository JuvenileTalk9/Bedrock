import json
import boto3
import botocore


def main() -> None:

    # パラメータ
    region_name = "us-east-1"
    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    # モデルに入力するプロンプト
    prompt = """
    あなたは優秀なAWSエンジニアです。AWSでサーバレスなWebAPIを実装するためのサービスとアーキテクチャ解説してください。
    """

    # モデルが要求するフォーマットに形成
    # Claude 3 Sonnet の場合のフォーマットは以下URLに記載
    # https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog/serverless/anthropic.claude-3-sonnet-20240229-v1:0
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        }
    )

    try:
        # クライアントを作成
        bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
        # 推論実行
        response = bedrock_client.invoke_model_with_response_stream(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )

        # 徐々に表示
        output = []
        i = 1
        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_obj = json.loads(chunk.get("bytes").decode())
                    if chunk_obj["type"] == "content_block_delta":
                        text = chunk_obj["delta"].get("text", "")
                        output.append(text)
                        print(f"\t\t\x1b[31m**Chunk {i}**\x1b[0m\n{text}\n")
                        i += 1
    except botocore.exceptions.ClientError as e:
        # 必要に応じてエラーハンドリング
        raise e


if __name__ == "__main__":
    main()
