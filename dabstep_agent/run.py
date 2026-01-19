from openai import OpenAI
import os
from agent import DataScienceAgent
from utils import solver
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("NVIDIA_API_KEY")


agent = DataScienceAgent(base_url = "https://integrate.api.nvidia.com/v1",
                         api_key = api_key,
                         max_iterations = 100,
                         model = "nvidia/nemotron-3-nano-30b-a3b",
                         verbose=True, stream=True, skip_final_response=False)

agent.reset_conversation()
#res = agent.process_prompt('find prime numbers within 200')
solver(0)
