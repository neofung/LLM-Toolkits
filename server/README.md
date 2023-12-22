Server side for LLM
=====================

## VLLM

### HOW TO RUN

1. Install dependencies

```shell
pip install fschat xformers vllm
```

2. Launch vllm

* [Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat)

```shell
python -m vllm.entrypoints.openai.api_server --model ./Yi-34B-Chat --trust-remote-code --host 0.0.0.0 --port 8082 \
  --dtype bfloat16  --tensor-parallel-size 2 --served-model-name yi-34b-chat
```

Fast test:

```shell
curl http://localhost:8082/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "yi-34b-chat",
"stop_token_ids": [7],
    "messages": [
      {
        "role": "user",
        "content": "你会什么？"
      }
    ],
"max_tokens": 512
}'
```

### TROUBLESHOOTING GUIDE

1. `urllib3` and `OpenSSL`

`ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: https://github.com/urllib3/urllib3/issues/2168`

> Or you could use an older version of urllib3 that is compatible suc. For example urllib3 v1.26.6,
> which does not have a strict OpenSSL version requirement. You can force the version installing with this command:

```bash
pip install urllib3==1.26.6
```

Ref:  [ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with LibreSSL 2.8.3](https://stackoverflow.com/questions/76187256/importerror-urllib3-v2-0-only-supports-openssl-1-1-1-currently-the-ssl-modu)

## Streamlit

### LAUNCH

```shell
streamlit run --server.address 0.0.0.0 --server.port 8081 ./streamlit_with_vllm.py 
```