import boto3
import ee
from botocore.client import Config as BotoConfig
from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GEE_PROJECT_ID: str
    OPENAI_API_KEY: str
    MAPBOX_API_KEY: str

    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_URL: str
    R2_BUCKET_NAME: str

    model_config = SettingsConfigDict(env_file=".env")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ee.Authenticate()
        ee.Initialize(project=self.GEE_PROJECT_ID)

    @property
    def async_openai_client(self):
        return AsyncOpenAI(api_key=self.OPENAI_API_KEY)

    @property
    def r2_client(self):
        """Return an R2-compatible boto3 client with SigV4 enforced."""

        return boto3.client(
            "s3",
            region_name="auto",  # Required by Cloudflare R2
            aws_access_key_id=self.R2_ACCESS_KEY_ID,
            aws_secret_access_key=self.R2_SECRET_ACCESS_KEY,
            endpoint_url=self.R2_URL,
            config=BotoConfig(signature_version="s3v4"),
        )


settings = Settings()
