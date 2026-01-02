import tritonclient.grpc.aio as grpcclient
import numpy as np
import json


class TritonClient:
    def __init__(self, url: str, model_name: str):
        self.client = grpcclient.InferenceServerClient(url)
        self.model_name = model_name
    
    async def inference(self, inputs, outputs):
        return await self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )


class LlamaInstructClient(TritonClient):

    async def generate(self, prompt: str, sampling_params: dict): 
        inputs = []
        inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=np.object_))

        inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(np.array([json.dumps(sampling_params).encode("utf-8")], dtype=np.object_))

        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([False], dtype=bool))

        async def request_generator():
            yield {"model_name": self.model_name, "inputs": inputs}

        response_iterator = self.client.stream_infer(inputs_iterator=request_generator())
        
        raw_response = ""
        async for response in response_iterator:
            result, error = response
            if error:
                raise error
            raw_response = result.as_numpy("text_output")[0].decode("utf-8")

        clean_text = self._clean_llama_response(raw_response)
        
        return clean_text
    
    def _clean_llama_response(self, raw_response: str) -> str:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in raw_response:
            return raw_response.split(marker)[-1].strip()
        return raw_response.strip()
