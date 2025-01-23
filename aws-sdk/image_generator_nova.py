import json
import base64

import cv2
import boto3
import botocore
import numpy as np


def main() -> None:

    # パラメータ
    region_name = "us-east-1"
    modelId = "amazon.nova-canvas-v1:0"
    accept = "application/json"
    contentType = "application/json"
    outputText = "n"

    # モデルに入力するプロンプト
    prompt = """
    A Japanese male system engineer wearing a suit, creating system design documents on a computer.
    """

    # モデルが要求するフォーマットに形成
    # Claude 3 Sonnet の場合のフォーマットは以下URLに記載
    # https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog/serverless/anthropic.claude-3-sonnet-20240229-v1:0
    body = json.dumps(
        {
            "textToImageParams": {"text": prompt},
            "taskType": "TEXT_IMAGE",
            "imageGenerationConfig": {
                "cfgScale": 8,
                "seed": 42,
                "quality": "standard",
                "width": 1280,
                "height": 720,
                "numberOfImages": 1,
            },
        }
    )

    try:
        # クライアントを作成
        bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
        # 推論実行
        response = bedrock_client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
    except botocore.exceptions.ClientError as e:
        # 必要に応じてエラーハンドリング
        raise e

    # 出力をパースして表示
    response_body = json.loads(response.get("body").read())
    outputText = response_body["images"][0]
    img_binary = base64.b64decode(outputText)
    img = cv2.imdecode(np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("debug/image_generator_nova.jpg", img)


if __name__ == "__main__":
    main()
