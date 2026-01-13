import asyncio

from nat.runtime.loader import load_workflow
from nat.utils.type_utils import StrPath
from dotenv import load_dotenv
from data_explorer_agent.utils import print_token_usage
load_dotenv()


async def run_workflow(config_file: StrPath, input_str: str) -> str:
    async with load_workflow(config_file) as workflow:
        async with workflow.run(input_str) as runner:
            return await runner.result()

if __name__ == "__main__":
    user_query='Based on this dataset in sample_data/QS_2025.csv, report the region that have the strongest education system. Make it a nice Notebook report.'
    result = asyncio.run(
        run_workflow(config_file='src/data_explorer_agent/configs/config.yml',
                        input_str=user_query))

    print("agent response:", result)
    print_token_usage("./traces.jsonl")