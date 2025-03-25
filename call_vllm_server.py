import requests

class VLLMClient:
    def __init__(self, server_url):
        self.server_url = server_url
        
    def generate(self, prompts, n=1, temperature=0.9, top_p=1.0, max_tokens=700, **kwargs):
        """Generate completions for prompts using vLLM server"""
        request_data = {
            "prompts": prompts,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        request_data.update(kwargs)
        
        response = requests.post(f"{self.server_url}/generate/", json=request_data)
        return response.json()
    
if __name__ == "__main__":
    client = VLLMClient("http://localhost:8000")
    print(client.generate(["Hello, how are you?"]))