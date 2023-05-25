# AutoRoute

## Env

### Isaac Gym

See [Isaac Gym](https://developer.nvidia.com/isaac-gym/download) to download, and run:
~~~bash
# prepare for Isaac Gym Env
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs/isaacgymenvs
pip install -e .
~~~

### LLMs
See [Vicuna](https://github.com/lm-sys/FastChat#vicuna-weights) to prepare its weight, and use [OpenAI Server](https://github.com/lm-sys/FastChat/blob/main/docs/langchain_integration.md#local-langchain-with-fastchat) to set up server.

Run the following comands to set the OpenAI API key:
~~~bash
export OPENAI_API_KEY=sk-xxx # your openai api key
~~~

## Training

~~~bash
cd src
python3 train.py task=QuadRoute headless=True
~~~

## Testing

~~~bash
cd src
python3 train.py task=QuadRoute checkpoint={path} test=True
~~~

## Paper

See [minipaper](doc/AutoRoute.pdf)