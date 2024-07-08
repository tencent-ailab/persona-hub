import argparse
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from code.prompt_templates import instruction_template, knowledge_template, npc_template, math_template

def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

def main(args):
    # Load the appropriate template
    if args.template == "instruction":
        template = instruction_template
    elif args.template == "knowledge":
        template = knowledge_template
    elif args.template == "npc":
        template = npc_template
    elif args.template == "math":
        template = math_template
    else:
        raise ValueError("Invalid template type. Choose from 'instruction', 'knowledge', 'math' or 'npc'.")

    # Load the dataset
    persona_dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")['train']
    if args.sample_size > 0:
        persona_dataset = persona_dataset[:args.sample_size]
    print(f"Total number of input personas: {len(persona_dataset['persona'])}")

    # Load the model and tokenizer
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=4) # please set tensor_parallel_size based on the GPUs you are using

    prompts = []
    max_len = 2048

    for persona in persona_dataset['persona']:
        persona = persona.strip()
        user_prompt = template.format(persona=persona)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_len, stop=["<|eot_id|>"])
    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, 'w') as out:
        for i, output in enumerate(outputs):
            out_txt = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            data = {'prompt': output.prompt, "input persona": persona_dataset['persona'][i].strip(), "finish_reason": finish_reason}
            data['synthesized text'] = out_txt
            out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')
    parser.add_argument(
        '--template', 
        type=str, 
        required=True, 
        choices=['instruction', 'knowledge', 'npc', 'math'], 
        help=(
            "Prompt templates. Choose from 'instruction', 'knowledge', 'math' or 'npc'. "
            "You can also add more customized templates in code/templates.py"
        )
    )
    args = parser.parse_args()
    main(args)
