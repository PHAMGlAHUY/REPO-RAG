from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = 'My API'
    db_url: str

    class Config:
        env_file = '.env'

settings = Settings()