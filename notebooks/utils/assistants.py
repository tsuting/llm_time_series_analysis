import pandas as pd
from openai import AzureOpenAI
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Union, Any
import time
from openai.types.beta.threads import TextContentBlock
from openai.types.beta.threads.runs import ToolCallsStepDetails
from pathlib import Path
import json
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.utils import convert_types
from utils.customized_func_tools import (  # noqa: F401
    get_time_col_and_target_col,
    get_descriptive_statistics,
    get_number_of_outliers,
    get_frequency,
    get_number_of_missing_datetime,
    get_number_of_null_values,
)

MODEL_ARGS = {
    "model": "gpt-4o",
    "temperature": 0,
    "top_p": 1,
}


class AzureOpenAIAssistant:
    def __init__(self, client: AzureOpenAI):
        self.client = client

    def list_all_assistants(self) -> pd.DataFrame:
        """List assistants in the client

        Returns:
            pd.DataFrame: assistants information
        """
        return pd.DataFrame.from_records(
            [s.to_dict() for s in self.client.beta.assistants.list().data]
        )

    def delete_assistant(self, name: str) -> None:
        """Delete assistants based on the name

        Args:
            name (str): the name of the assistant
        """
        df = self.list_all_assistants()
        if not df.empty:
            for id in df[df["name"] == name]["id"].tolist():
                self.client.beta.assistants.delete(assistant_id=id)

    def list_all_files(self) -> pd.DataFrame:
        """List files in the client

        Returns:
            pd.DataFrame: file information
        """
        return pd.DataFrame.from_records(
            [s.to_dict() for s in self.client.files.list().data]
        )

    def delete_a_file(self, file_name: str) -> None:
        """Delete a file in the client

        Args:
            file_name (str): file name to be deleted
        """
        df = self.list_all_files()
        for id in df[df["filename"] == file_name]["id"].tolist():
            self.client.files.delete(file_id=id)

    def upload_a_file(self, file_path) -> str:
        """Upload a file to the client

        Args:
            file_path (str): the path of the file

        Returns:
            str: the file id
        """
        with open(file_path, "rb") as file:
            return self.client.files.create(file=file, purpose="assistants").id

    def upload_or_retrieve_file(self, file_path: str) -> str:
        """upload or retrieve a file id based on the file path

        If the file name has been uploaded, it does not upload the file again, it uploads the file and returns the file id.

        Args:
            file_path (str): file path

        Returns:
            str: file id
        """
        file_id = None
        df_files = self.list_all_files()

        if not df_files.empty:
            df_file = df_files[df_files["filename"] == file_path.name]
            if not df_file.empty:
                file_id = df_file.iloc[0]["id"]
        if not file_id:
            file_id = self.upload_a_file(file_path=file_path)

        return file_id

    def create_or_retrieve(
        self,
        assistant_name: str,
        prompt_path: str = None,
        tools: Dict[str, Union[str, Dict[str, Any]]] = {},
        tool_resources: Dict[str, Dict[str, str]] = {},
        prompt_args: Union[Dict[str, str]] = {},
    ) -> str:
        """Create or retrieve an assistant with the given name and return assistant id

        Args:
            assistant_name (str): the name of assistant
            prompt_path (str, optional): the path of the prompt to read. Defaults to None.
            tools (Dict[str, Union[str, Dict[str, Any]]], optional): the tools to use for the assistant. Defaults to {}.
            tool_resources (Dict[str, Dict[str, str]], optional): the resources for the tools. Defaults to {}.
            prompt_args (Union[Dict[str, str], None], optional): the arguments for rendering the prompt. Defaults to {}.

        Returns:
            str: assistant id
        """
        # get the list of assistants
        df_assistants = self.list_all_assistants()

        if not df_assistants.empty:
            df_assistants = df_assistants[df_assistants["name"] == assistant_name]

        if df_assistants.shape[0] == 0:
            # read the prompt
            instruction = (
                Environment(loader=FileSystemLoader("."))
                .get_template(prompt_path)
                .render(**prompt_args)
            )

            # create an assistant
            assistant = self.client.beta.assistants.create(
                **{
                    **MODEL_ARGS,
                    **{
                        "name": assistant_name,
                        "instructions": instruction,
                        "tools": tools,
                        "tool_resources": tool_resources,
                    },
                }
            )
        elif df_assistants.shape[0] == 1:
            assistant_id = df_assistants.iloc[0]["id"]
            assistant = self.client.beta.assistants.retrieve(assistant_id=assistant_id)
        else:
            raise ValueError("More than one assistant with the same name.")

        return assistant.id

    def ask_a_question(
        self,
        question: str,
        assistant_id: str,
        tools: Dict[str, Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Ask a question to the assistant and return the response

        Args:
            question (str): the question to ask
            assistant_id (str): the id of the assistant
            tools (Dict[str, Union[str, Dict[str, Any]]], optional): the dictionary defines all the available functions. Defaults to None.
                This will only be used for customized func calling not code interpreter.

        Raises:
            ValueError: when the function does not exist
            ValueError: when the run status is unexpected

        Returns:
            Dict[str, Any]: the response
                it includes the following keys:
                    completion_tokens, prompt_tokens, total_tokens, question, answer_pred, message, code, attachment, execution_time_s
        """
        # create a and run a thread
        run = self.client.beta.threads.create_and_run(
            assistant_id=assistant_id,
            thread={"messages": [{"role": "user", "content": question}]},
        )

        # looping until the run completes or fails
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=run.thread_id, run_id=run.id
            )
            match run.status:
                case "completed":
                    messages = self.client.beta.threads.messages.list(
                        thread_id=run.thread_id
                    )

                    # format the output
                    # for message
                    result_messages = []
                    # for attachment if there is any
                    result_attachments = []

                    # sort the order from old to new
                    for message in messages.data[::-1]:
                        for item in message.content:
                            if isinstance(item, TextContentBlock):
                                # add to message
                                result_messages.append(
                                    f"{message.role}: {item.text.value}"
                                )

                                # if the text got attachments
                                if item.text.annotations != []:
                                    result_attachments.append(
                                        {
                                            "file_bytes": self.client.files.content(
                                                item.text.annotations[
                                                    0
                                                ].file_path.file_id
                                            ).read(),
                                            "file_name": Path(
                                                item.text.annotations[0].text
                                            ).name,
                                        }
                                    )

                    # get answer_pred; the output is {"'assistant': {'output': 'the answer'}"}
                    answer_pred = result_messages[-1].split(": ", 1)[-1]
                    try:
                        answer_pred = json.loads(answer_pred)["output"]
                    except json.decoder.JSONDecodeError:
                        print(f"JSONDecodeError: {answer_pred}")

                    # get the steps to call functions
                    result_steps = []
                    run_steps = self.client.beta.threads.runs.steps.list(
                        thread_id=run.thread_id, run_id=run.id
                    )

                    # sort the order from old to new
                    for step in run_steps.data[::-1]:
                        if isinstance(step.step_details, ToolCallsStepDetails):
                            if len(step.step_details.tool_calls) != 1:
                                print("Weird in step_details.")
                            else:
                                tool_call = step.step_details.tool_calls[0]
                                if tool_call.type == "function":
                                    result_steps.append(
                                        {
                                            "name": tool_call.function.name,
                                            "input": None,
                                            "output": tool_call.function.output,
                                            "args": tool_call.function.arguments,
                                        }
                                    )
                                else:
                                    result_steps.append(
                                        {
                                            "name": "code_interpreter",
                                            "input": tool_call.code_interpreter.input,
                                            "output": None,
                                            "args": None,
                                        }
                                    )

                    return {
                        # get prompts usage
                        **run.usage.to_dict(),
                        "question": question,
                        "answer_pred": convert_types(answer_pred),
                        "messages": result_messages,
                        "steps": result_steps,
                        "attachments": result_attachments,
                        "execution_time_s": run.completed_at - run.created_at,
                    }

                case "requires_action":
                    # the assistant requires calling some functions
                    # and submit the tool outputs back to the run

                    available_functions = {
                        func_name: globals()[func_name]
                        for func_name in pd.json_normalize(tools)
                        .query("type == 'function'")["function.name"]
                        .tolist()
                    }
                    tool_responses = []
                    if (
                        run.required_action.type == "submit_tool_outputs"
                        and run.required_action.submit_tool_outputs.tool_calls
                        is not None
                    ):
                        tool_calls = run.required_action.submit_tool_outputs.tool_calls

                        for call in tool_calls:
                            if call.type == "function":
                                if call.function.name not in available_functions:
                                    raise ValueError(
                                        f"Function requested by the model does not exist: {call.function.name}"
                                    )

                                function_to_call = available_functions[
                                    call.function.name
                                ]
                                tool_response = function_to_call(
                                    **json.loads(call.function.arguments)
                                )
                                tool_responses.append(
                                    {
                                        "tool_call_id": call.id,
                                        "output": str(tool_response),
                                    }
                                )

                    run = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=run.thread_id,
                        run_id=run.id,
                        tool_outputs=tool_responses,
                    )
                case "queued" | "in_progress":
                    time.sleep(1)
                case _:
                    raise ValueError(f"Unexpected run status: {run.status}")
